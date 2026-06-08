"""Tests for resilient progress reporting against a flaky Audiobookshelf connection.

Playback is served from buffer and continues even when MA cannot reach abs, so a failed
progress write must be retained (not silently dropped) and surfaced again on resume.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from aioaudiobookshelf.exceptions import SessionSyncError
from aiohttp import ClientError
from music_assistant_models.enums import MediaType
from music_assistant_models.media_items import Audiobook

from music_assistant.constants import PLAYBACK_REPORT_INTERVAL_SECONDS
from music_assistant.providers.audiobookshelf import Audiobookshelf
from music_assistant.providers.audiobookshelf.helpers import PendingProgress, ProgressOutbox

BOOK_ID = "book1"
DURATION = 3600


def _bind(provider: Mock, *names: str) -> None:
    """Bind the real (unbound) Audiobookshelf methods onto a spec mock."""
    for name in names:
        setattr(provider, name, getattr(Audiobookshelf, name).__get__(provider, Audiobookshelf))


@pytest.fixture
def provider() -> Mock:
    """Return an Audiobookshelf spec mock wired for progress reporting."""
    provider = Mock(spec=Audiobookshelf)
    provider.logger = Mock()
    provider.sessions = {}
    provider._client = AsyncMock()
    provider.progress_guard = Mock()
    provider.progress_guard.guard_ok_mass.return_value = True
    provider.progress_outbox = ProgressOutbox()
    _bind(
        provider,
        "on_played",
        "_sync_progress_to_abs",
        "_update_by_session",
        "_flush_pending",
        "get_resume_position",
    )
    return provider


def _audiobook() -> Mock:
    media_item = Mock(spec=Audiobook)
    media_item.duration = DURATION
    media_item.name = "Some Book"
    return media_item


@pytest.mark.asyncio
async def test_failed_sync_is_retained_in_outbox(provider: Mock) -> None:
    """A transport error during the abs write is swallowed but the position is kept."""
    provider._client.update_my_media_progress.side_effect = ClientError("connection reset")

    # must not raise: progress reporting is advisory and playback is unaffected
    await provider.on_played(
        MediaType.AUDIOBOOK, BOOK_ID, fully_played=False, position=1500, media_item=_audiobook()
    )

    provider._client.update_my_media_progress.assert_awaited_once()
    pending = provider.progress_outbox.get(BOOK_ID)
    assert pending is not None
    assert pending.position == 1500
    assert pending.fully_played is False


@pytest.mark.asyncio
async def test_successful_sync_clears_outbox(provider: Mock) -> None:
    """A confirmed abs write leaves nothing pending."""
    await provider.on_played(
        MediaType.AUDIOBOOK, BOOK_ID, fully_played=False, position=1500, media_item=_audiobook()
    )

    provider._client.update_my_media_progress.assert_awaited_once()
    assert provider.progress_outbox.get(BOOK_ID) is None


@pytest.mark.asyncio
async def test_resume_uses_retained_progress_when_abs_stale(provider: Mock) -> None:
    """When abs is missing our latest update, resume returns the retained position."""
    provider._get_playback_session = AsyncMock(
        return_value=Mock(current_time=100.0, duration=DURATION)
    )
    # abs only knows the session-start position, last updated long ago (epoch seconds → ms)
    provider._client.get_my_media_progress.return_value = Mock(
        current_time=100.0, last_update=1000 * 1000
    )
    provider.progress_outbox.record(BOOK_ID, position=1500, duration=DURATION, fully_played=False)

    finished, position_ms, _ = await provider.get_resume_position(BOOK_ID, MediaType.AUDIOBOOK)

    assert position_ms == 1500 * 1000
    assert finished is False
    # read-repair: we try to push the retained value, and clear it once abs accepts
    provider._client.update_my_media_progress.assert_awaited_once()
    assert provider.progress_outbox.get(BOOK_ID) is None


@pytest.mark.asyncio
async def test_resume_prefers_fresher_rewind_over_higher_abs_position(provider: Mock) -> None:
    """Freshness, not max position, wins: a retained rewind beats a higher abs position."""
    provider._get_playback_session = AsyncMock(
        return_value=Mock(current_time=1500.0, duration=DURATION)
    )
    provider._client.get_my_media_progress.return_value = Mock(
        current_time=1500.0, last_update=1000 * 1000
    )
    # user rewound to 600s while the connection was flaky; this is the freshest intent
    provider.progress_outbox.record(BOOK_ID, position=600, duration=DURATION, fully_played=False)

    _, position_ms, _ = await provider.get_resume_position(BOOK_ID, MediaType.AUDIOBOOK)

    assert position_ms == 600 * 1000


@pytest.mark.asyncio
async def test_resume_prefers_abs_when_it_is_fresher(provider: Mock) -> None:
    """An external/newer abs update supersedes a stale retained entry, which is dropped."""
    provider._get_playback_session = AsyncMock(
        return_value=Mock(current_time=2000.0, duration=DURATION)
    )
    # abs last_update is far newer than the retained entry below
    provider._client.get_my_media_progress.return_value = Mock(
        current_time=2000.0, last_update=9_000_000 * 1000
    )
    provider.progress_outbox._entries[BOOK_ID] = PendingProgress(
        position=600, duration=DURATION, fully_played=False, updated_at=1000.0
    )

    _, position_ms, _ = await provider.get_resume_position(BOOK_ID, MediaType.AUDIOBOOK)

    assert position_ms == 2000 * 1000
    # superseded pending is cleared and no read-repair write is attempted
    assert provider.progress_outbox.get(BOOK_ID) is None
    provider._client.update_my_media_progress.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_by_session_falls_back_on_transport_error(provider: Mock) -> None:
    """A session sync transport error returns False so the direct update is attempted."""
    provider._client.sync_open_session.side_effect = SessionSyncError()
    session_helper = Mock(abs_session_id="sess", last_sync_time=0.0)

    result = await provider._update_by_session(
        session_helper=session_helper, position=1500, duration=DURATION
    )

    assert result is False


def test_compare_and_clear_keeps_newer_entry() -> None:
    """A stale write clearing by identity must not wipe a newer report's entry."""
    outbox = ProgressOutbox()
    older = outbox.record(BOOK_ID, position=1500, duration=DURATION, fully_played=False)
    newer = outbox.record(BOOK_ID, position=1530, duration=DURATION, fully_played=False)

    # the older (slow) write completes last and tries to clear its own entry
    outbox.clear(BOOK_ID, older)
    assert outbox.get(BOOK_ID) is newer

    # the newer write's clear does remove it
    outbox.clear(BOOK_ID, newer)
    assert outbox.get(BOOK_ID) is None


@pytest.mark.asyncio
async def test_mark_unplayed_discards_retained_progress(provider: Mock) -> None:
    """Marking unplayed must drop any retained progress, even when abs has none."""
    provider._client.get_my_media_progress.return_value = None  # abs has no progress
    provider.progress_outbox.record(BOOK_ID, position=1500, duration=DURATION, fully_played=False)

    await provider.on_played(
        MediaType.AUDIOBOOK, BOOK_ID, fully_played=False, position=0, media_item=_audiobook()
    )

    assert provider.progress_outbox.get(BOOK_ID) is None


@pytest.mark.asyncio
async def test_external_socket_update_invalidates_retained_progress(provider: Mock) -> None:
    """A genuinely external abs update supersedes and clears our retained entry."""
    provider._get_all_known_item_ids = Mock(return_value={BOOK_ID})
    provider._update_playlog_book = AsyncMock()
    provider.progress_guard.guard_ok_abs.return_value = True
    provider.progress_outbox.record(BOOK_ID, position=1500, duration=DURATION, fully_played=False)
    _bind(provider, "_socket_abs_user_item_progress_updated")

    external = Mock(library_item_id=BOOK_ID, episode_id=None, current_time=200.0)
    await provider._socket_abs_user_item_progress_updated(BOOK_ID, external)

    assert provider.progress_outbox.get(BOOK_ID) is None
    provider._update_playlog_book.assert_awaited_once()


@pytest.mark.asyncio
async def test_faulty_full_played_position_is_ignored(provider: Mock) -> None:
    """A fully_played report with an implausibly low position is dropped, not synced."""
    await provider.on_played(
        MediaType.AUDIOBOOK,
        BOOK_ID,
        fully_played=True,
        position=DURATION - PLAYBACK_REPORT_INTERVAL_SECONDS - 100,
        media_item=_audiobook(),
    )

    provider._client.update_my_media_progress.assert_not_awaited()
    assert provider.progress_outbox.get(BOOK_ID) is None
