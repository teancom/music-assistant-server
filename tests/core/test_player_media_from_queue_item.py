"""Tests for player_media_from_queue_item in PlayerQueuesController."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import ContentType, ImageType, MediaType, StreamType
from music_assistant_models.media_items import AudioFormat, MediaItemImage, ProviderMapping, Radio
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import StreamDetails, StreamMetadata

from music_assistant.constants import MASS_LOGO_ONLINE
from music_assistant.controllers.player_queues import PlayerQueuesController

QUEUE_ID = "test_queue"
SESSION_ID = "test_session"
STATIC_IMAGE_URL = "http://mass.local/imageproxy?path=station_logo.png&provider=radio&size=500"
STREAM_PROXIED_URL = "http://mass.local/imageproxy?path=stream_cover.jpg&provider=url&size=500"
STREAM_IMAGE_URL = "https://img.radioparadise.com/covers/l/19806.jpg"


def _make_controller(queue: PlayerQueue) -> PlayerQueuesController:
    """Create a minimal PlayerQueuesController with mocked internals."""
    ctrl = object.__new__(PlayerQueuesController)
    ctrl._queues = {queue.queue_id: queue}

    # Mock mass.metadata.get_image_url to return different URLs based on provider
    mass = MagicMock()

    def _mock_get_image_url(image, **_kwargs):
        if image.provider == "url":
            return STREAM_PROXIED_URL
        return STATIC_IMAGE_URL

    mass.metadata.get_image_url.side_effect = _mock_get_image_url
    ctrl.mass = mass

    return ctrl


def _make_queue() -> PlayerQueue:
    """Create a minimal PlayerQueue."""
    return PlayerQueue(
        queue_id=QUEUE_ID,
        active=True,
        display_name="Test Queue",
        available=True,
        items=1,
        session_id=SESSION_ID,
    )


def _make_radio_media_item() -> Radio:
    """Create a minimal Radio media item."""
    return Radio(
        provider="radioparadise",
        item_id="main",
        name="Radio Paradise - Main Mix",
        provider_mappings={
            ProviderMapping(
                provider_domain="radioparadise",
                provider_instance="radioparadise",
                item_id="main",
                available=True,
            )
        },
    )


def _make_queue_item(
    *,
    stream_metadata: StreamMetadata | None = None,
    has_image: bool = False,
    has_media_item: bool = True,
    has_streamdetails: bool = True,
) -> QueueItem:
    """Create a QueueItem with configurable stream metadata and image."""
    streamdetails = None
    if has_streamdetails:
        streamdetails = StreamDetails(
            provider="radioparadise",
            item_id="main",
            audio_format=AudioFormat(content_type=ContentType.AAC),
            media_type=MediaType.RADIO,
            stream_type=StreamType.HTTP,
            stream_metadata=stream_metadata,
        )

    image = None
    if has_image:
        image = MediaItemImage(
            provider="radioparadise",
            type=ImageType.THUMB,
            path="https://example.com/station_logo.png",
            remotely_accessible=True,
        )

    return QueueItem(
        queue_id=QUEUE_ID,
        queue_item_id="item_001",
        name="Radio Paradise - Main Mix",
        duration=0,
        streamdetails=streamdetails,
        media_item=_make_radio_media_item() if has_media_item else None,
        image=image,
    )


@pytest.mark.asyncio
async def test_stream_metadata_image_overrides_static_image() -> None:
    """Stream metadata image_url (current track art) should override the static media item image."""
    queue = _make_queue()
    ctrl = _make_controller(queue)

    stream_metadata = StreamMetadata(
        title="Lay It Down",
        artist="Cowboy Junkies",
        album="Lay It Down (2008)",
        image_url=STREAM_IMAGE_URL,
    )
    queue_item = _make_queue_item(stream_metadata=stream_metadata, has_image=True)

    media = await ctrl.player_media_from_queue_item(queue_item)

    # Should use the proxied stream metadata image, not the static station logo
    assert media.image_url == STREAM_PROXIED_URL
    # Verify the stream image was proxied with correct parameters
    last_call = ctrl.mass.metadata.get_image_url.call_args_list[-1]
    stream_image_arg = last_call[0][0]
    assert stream_image_arg.path == STREAM_IMAGE_URL
    assert stream_image_arg.provider == "url"
    assert last_call[1]["size"] == 500
    assert last_call[1]["prefer_stream_server"] is True


@pytest.mark.asyncio
async def test_static_image_used_when_no_stream_metadata() -> None:
    """Without stream metadata, the static media item image should be used."""
    queue = _make_queue()
    ctrl = _make_controller(queue)

    queue_item = _make_queue_item(stream_metadata=None, has_image=True)

    media = await ctrl.player_media_from_queue_item(queue_item)

    assert media.image_url == STATIC_IMAGE_URL


@pytest.mark.asyncio
async def test_static_image_used_when_stream_metadata_has_no_image() -> None:
    """Stream metadata without image_url should fall back to static image."""
    queue = _make_queue()
    ctrl = _make_controller(queue)

    stream_metadata = StreamMetadata(
        title="Some Song",
        artist="Some Artist",
        image_url=None,
    )
    queue_item = _make_queue_item(stream_metadata=stream_metadata, has_image=True)

    media = await ctrl.player_media_from_queue_item(queue_item)

    assert media.image_url == STATIC_IMAGE_URL


@pytest.mark.asyncio
async def test_stream_metadata_image_used_even_without_static_image() -> None:
    """Stream metadata image should work even when there's no static media item image."""
    queue = _make_queue()
    ctrl = _make_controller(queue)

    stream_metadata = StreamMetadata(
        title="Lay It Down",
        artist="Cowboy Junkies",
        image_url=STREAM_IMAGE_URL,
    )
    queue_item = _make_queue_item(stream_metadata=stream_metadata, has_image=False)

    media = await ctrl.player_media_from_queue_item(queue_item)

    # Should use the proxied stream metadata image
    assert media.image_url == STREAM_PROXIED_URL


@pytest.mark.asyncio
async def test_fallback_to_mass_logo_when_no_images() -> None:
    """Without any image source, should fall back to MASS_LOGO_ONLINE."""
    queue = _make_queue()
    ctrl = _make_controller(queue)

    queue_item = _make_queue_item(stream_metadata=None, has_image=False)

    media = await ctrl.player_media_from_queue_item(queue_item)

    assert media.image_url == MASS_LOGO_ONLINE


@pytest.mark.asyncio
async def test_no_streamdetails_with_static_image() -> None:
    """A regular (non-radio) queue item with an image but no streamdetails uses static image."""
    queue = _make_queue()
    ctrl = _make_controller(queue)

    queue_item = _make_queue_item(has_streamdetails=False, has_image=True)

    media = await ctrl.player_media_from_queue_item(queue_item)

    # No streamdetails means no stream_metadata override, so static image is used.
    assert media.image_url == STATIC_IMAGE_URL
