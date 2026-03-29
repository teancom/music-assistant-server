"""Shared test fixtures for Overcast provider tests."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from music_assistant.providers.overcast.constants import CONF_COOKIE
from music_assistant.providers.overcast.provider import OvercastProvider


@pytest.fixture
def mass_mock() -> MagicMock:
    """Return a mock MusicAssistant instance."""
    mass = MagicMock()
    mass.http_session = AsyncMock()
    mass.cache = AsyncMock()
    return mass


@pytest.fixture
def manifest_mock() -> MagicMock:
    """Return a mock provider manifest."""
    manifest = MagicMock()
    manifest.domain = "overcast"
    return manifest


@pytest.fixture
def config_mock_with_cookie() -> MagicMock:
    """Return a mock provider config with valid cookie."""
    config = MagicMock()
    config.name = "Overcast Test"
    config.instance_id = "overcast_test"
    config.enabled = True
    config.get_value.side_effect = lambda key, default=None: {
        CONF_COOKIE: "valid_cookie_value",
        "log_level": "INFO",
    }.get(key, default)
    return config


@pytest.fixture
def config_mock_no_cookie() -> MagicMock:
    """Return a mock provider config without cookie."""
    config = MagicMock()
    config.name = "Overcast Test"
    config.instance_id = "overcast_test"
    config.enabled = True
    config.get_value.side_effect = lambda key, default=None: {
        "log_level": "INFO",
    }.get(key, default)
    return config


@pytest.fixture
def sample_opml_data() -> dict[str, Any]:
    """
    Return parsed OPML data for testing.

    Fixture includes 2 podcasts with episodes having various play states
    (played, progress, unplayed) to test all scenarios.
    """
    return {
        "podcasts": [
            {
                "overcast_id": "2001",
                "title": "Podcast One",
                "xml_url": "https://example.com/podcast1.xml",
                "html_url": "https://example.com/podcast1",
                "episodes": [
                    {
                        "overcast_id": "3001",
                        "title": "Episode 1",
                        "pub_date": "2024-10-12T16:00:00-04:00",
                        "enclosure_url": "https://example.com/ep1.mp3",
                        "overcast_url": "https://overcast.fm/+ABC123",
                        "played": "1",
                        "progress": None,
                        "user_deleted": None,
                    },
                    {
                        "overcast_id": "3002",
                        "title": "Episode 2",
                        "pub_date": "2024-10-11T16:00:00-04:00",
                        "enclosure_url": "https://example.com/ep2.mp3",
                        "overcast_url": "https://overcast.fm/+ABC124",
                        "played": None,
                        "progress": "120",
                        "user_deleted": None,
                    },
                    {
                        "overcast_id": "3003",
                        "title": "Episode 3",
                        "pub_date": "2024-10-10T16:00:00-04:00",
                        "enclosure_url": "https://example.com/ep3.mp3",
                        "overcast_url": "https://overcast.fm/+ABC125",
                        "played": None,
                        "progress": None,
                        "user_deleted": None,
                    },
                    {
                        "overcast_id": "3004",
                        "title": "Episode 4",
                        "pub_date": "2024-10-09T16:00:00-04:00",
                        "enclosure_url": None,
                        "overcast_url": "https://overcast.fm/+ABC126",
                        "played": None,
                        "progress": None,
                        "user_deleted": None,
                    },
                ],
            },
            {
                "overcast_id": "2002",
                "title": "Podcast Two",
                "xml_url": "https://example.com/podcast2.xml",
                "html_url": "https://example.com/podcast2",
                "episodes": [
                    {
                        "overcast_id": "4001",
                        "title": "Episode A",
                        "pub_date": "2024-10-08T16:00:00-04:00",
                        "enclosure_url": "https://example.com/epA.mp3",
                        "overcast_url": "https://overcast.fm/+XYZ789",
                        "played": None,
                        "progress": None,
                        "user_deleted": None,
                    },
                    {
                        "overcast_id": "4002",
                        "title": "Episode B",
                        "pub_date": "2024-10-07T16:00:00-04:00",
                        "enclosure_url": "https://example.com/epB.mp3",
                        "overcast_url": "https://overcast.fm/+XYZ790",
                        "played": "1",
                        "progress": None,
                        "user_deleted": None,
                    },
                ],
            },
        ],
        "playlists": [
            {
                "title": "All Episodes",
                "smart": "1",
                "episode_ids": ["3001", "3002", "3003", "4001", "4002"],
            },
            {
                "title": "Queue",
                "smart": "0",
                "episode_ids": ["3002", "4001"],
            },
        ],
    }


@pytest.fixture
async def provider_with_mocked_data(
    mass_mock: MagicMock,
    manifest_mock: MagicMock,
    config_mock_with_cookie: MagicMock,
    sample_opml_data: dict[str, Any],
) -> OvercastProvider:
    """
    Return OvercastProvider with mocked _get_opml_data() returning test data.

    This fixture creates a provider instance and patches _get_opml_data()
    to return pre-parsed OPML data from the sample fixture, avoiding
    actual HTTP calls in tests.
    """
    provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

    async def mock_get_opml_data() -> dict[str, Any]:
        return sample_opml_data

    provider._get_opml_data = mock_get_opml_data  # type: ignore[method-assign]

    async def mock_handle_async_init() -> None:
        cookie = config_mock_with_cookie.get_value(CONF_COOKIE)
        provider._cookie = str(cookie) if cookie else ""
        # Per-instance dicts so test classes don't share state.
        provider._sync_versions = {}
        provider._last_progress_time = {}

    provider.handle_async_init = mock_handle_async_init  # type: ignore[method-assign]

    await provider.handle_async_init()

    return provider
