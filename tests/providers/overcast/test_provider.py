"""Test Overcast Provider integration."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.errors import (
    LoginFailed,
    MediaNotFoundError,
    MusicAssistantError,
    RetriesExhausted,
)
from music_assistant_models.media_items import BrowseFolder, PodcastEpisode

from music_assistant.helpers.throttle_retry import ThrottlerManager
from music_assistant.providers.overcast.constants import (
    BROWSE_PLAYLISTS,
    BROWSE_PODCASTS,
    BROWSE_UNPLAYED,
    CACHE_CATEGORY_OPML,
    CACHE_KEY_OPML,
    CACHE_KEY_RSS_ENRICHMENT,
    CACHE_TTL,
    CACHE_TTL_ENRICHMENT,
    FINISHED_SENTINEL,
    GUARD_INTERVAL_ACTION,
    GUARD_INTERVAL_PLAYING,
)
from music_assistant.providers.overcast.provider import OvercastProvider


async def test_handle_async_init_with_valid_cookie(
    mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
) -> None:
    """
    Test successful async initialization with valid cookie (AC1.1).

    Valid session cookie accepted, provider initializes successfully.
    """
    provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

    # Mock successful HTTP response
    response = AsyncMock()
    response.status = 200
    opml_content = "<opml version='1.0'><body><outline text='feeds'/></body></opml>"
    response.text = AsyncMock(return_value=opml_content)

    # Create a mock context manager for the HTTP request
    request_ctx = AsyncMock()
    request_ctx.__aenter__.return_value = response
    request_ctx.__aexit__.return_value = None

    mass_mock.http_session.get = MagicMock(return_value=request_ctx)
    mass_mock.cache.get = AsyncMock(return_value=None)
    mass_mock.cache.set = AsyncMock()

    # Should not raise any exception
    await provider.handle_async_init()

    # Verify the cookie was stored
    assert provider._cookie == "valid_cookie_value"

    # Verify the HTTP request was made correctly
    mass_mock.http_session.get.assert_called_once()
    call_args = mass_mock.http_session.get.call_args
    assert "headers" in call_args.kwargs
    assert call_args.kwargs["headers"]["Cookie"] == "o=valid_cookie_value"


async def test_handle_async_init_with_invalid_cookie(
    mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
) -> None:
    """
    Test async initialization with invalid/expired cookie (AC1.2).

    Invalid/expired cookie raises LoginFailed with message to update cookie.
    """
    provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

    # Mock 401 response (unauthorized)
    response = AsyncMock()
    response.status = 401

    request_ctx = AsyncMock()
    request_ctx.__aenter__.return_value = response
    request_ctx.__aexit__.return_value = None

    mass_mock.http_session.get = MagicMock(return_value=request_ctx)
    mass_mock.cache.get = AsyncMock(return_value=None)

    # Should raise LoginFailed
    with pytest.raises(LoginFailed) as exc_info:
        await provider.handle_async_init()

    # Verify error message mentions updating cookie or expiry
    assert "expired" in str(exc_info.value).lower()


async def test_handle_async_init_with_empty_cookie(
    mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_no_cookie: MagicMock
) -> None:
    """
    Test async initialization with empty/missing cookie (AC1.3).

    Empty cookie config raises LoginFailed.
    """
    provider = OvercastProvider(mass_mock, manifest_mock, config_mock_no_cookie)

    # Should raise LoginFailed before making HTTP request
    with pytest.raises(LoginFailed) as exc_info:
        await provider.handle_async_init()

    # Verify error message mentions providing cookie
    assert "required" in str(exc_info.value).lower()

    # Verify HTTP request was NOT made
    mass_mock.http_session.get.assert_not_called()


async def test_handle_async_init_with_redirect_to_login(
    mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
) -> None:
    """
    Test async initialization with redirect to login (AC1.2 variant).

    Cookie that causes redirect to login raises LoginFailed.
    """
    provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

    # Mock redirect response (302) to login page
    response = AsyncMock()
    response.status = 302
    response.headers = {"Location": "/login"}

    request_ctx = AsyncMock()
    request_ctx.__aenter__.return_value = response
    request_ctx.__aexit__.return_value = None

    mass_mock.http_session.get = MagicMock(return_value=request_ctx)
    mass_mock.cache.get = AsyncMock(return_value=None)

    # Should raise LoginFailed
    with pytest.raises(LoginFailed) as exc_info:
        await provider.handle_async_init()

    # Verify error message mentions updating cookie or expiry
    assert "expired" in str(exc_info.value).lower()


class TestLibrarySyncMethods:
    """Tests for library sync methods (AC2)."""

    async def test_get_library_podcasts_yields_correct_count(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Get library podcasts returns correct number of Podcast items (AC2.1).

        get_library_podcasts() should yield all subscribed podcasts.
        """
        podcasts = []
        async for podcast in provider_with_mocked_data.get_library_podcasts():
            podcasts.append(podcast)

        # Sample data has 2 podcasts
        assert len(podcasts) == 2

    async def test_get_library_podcasts_have_required_fields(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Each Podcast has correct fields and provider mapping (AC2.2).

        Podcasts should have item_id (overcastId), name, and RSS URL in mapping.
        """
        podcasts = []
        async for podcast in provider_with_mocked_data.get_library_podcasts():
            podcasts.append(podcast)

        podcast = podcasts[0]

        # Check AC2.2 - podcast has required fields
        assert podcast.item_id == "2001"
        assert podcast.name == "Podcast One"
        assert podcast.provider == "overcast_test"

        # Check provider mapping has XML URL
        assert len(podcast.provider_mappings) == 1
        mapping = next(iter(podcast.provider_mappings))
        assert mapping.url == "https://example.com/podcast1.xml"
        assert mapping.item_id == "2001"

    async def test_get_podcast_returns_correct_podcast(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Get podcast returns correct Podcast by overcastId."""
        podcast = await provider_with_mocked_data.get_podcast("2001")

        assert podcast.item_id == "2001"
        assert podcast.name == "Podcast One"

    async def test_get_podcast_raises_for_unknown_id(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Get podcast raises MediaNotFoundError for unknown ID."""
        with pytest.raises(MediaNotFoundError):
            await provider_with_mocked_data.get_podcast("9999")

    async def test_get_podcast_episodes_yields_correct_count(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Get podcast episodes yields correct number of episodes (AC2.3).

        Should yield only episodes with enclosure URL, skipping others.
        """
        episodes = []
        async for episode in provider_with_mocked_data.get_podcast_episodes("2001"):
            episodes.append(episode)

        # Podcast 2001 has 4 episodes in fixture, but one has no enclosureUrl
        # So should yield 3 episodes
        assert len(episodes) == 3

    async def test_get_podcast_episodes_have_required_fields(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Each PodcastEpisode has required fields (AC2.3)."""
        episodes = []
        async for episode in provider_with_mocked_data.get_podcast_episodes("2001"):
            episodes.append(episode)

        episode = episodes[0]

        # Check AC2.3 - episode has required fields
        assert episode.item_id == "3001"
        assert episode.name == "Episode 1"
        assert episode.provider == "overcast_test"

        # Check provider mapping has enclosure URL
        assert len(episode.provider_mappings) == 1
        mapping = next(iter(episode.provider_mappings))
        assert mapping.url == "https://example.com/ep1.mp3"
        assert mapping.item_id == "3001"

    async def test_get_podcast_episodes_sets_fully_played(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Episodes with played='1' have fully_played=True (AC2.4)."""
        episodes = []
        async for episode in provider_with_mocked_data.get_podcast_episodes("2001"):
            episodes.append(episode)

        # Episode 3001 has played="1"
        played_episode = episodes[0]
        assert played_episode.item_id == "3001"
        assert played_episode.fully_played is True

    async def test_get_podcast_episodes_sets_resume_position(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Episodes with progress set resume_position_ms (AC2.4).

        Progress in seconds should be converted to milliseconds.
        """
        episodes = []
        async for episode in provider_with_mocked_data.get_podcast_episodes("2001"):
            episodes.append(episode)

        # Episode 3002 has progress="120"
        resume_episode = episodes[1]
        assert resume_episode.item_id == "3002"
        assert resume_episode.resume_position_ms == 120000

    async def test_get_podcast_episodes_unplayed_has_no_resume_position(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Episodes without play state have None for play fields (AC2.4)."""
        episodes = []
        async for episode in provider_with_mocked_data.get_podcast_episodes("2001"):
            episodes.append(episode)

        # Episode 3003 has no played or progress
        unplayed_episode = episodes[2]
        assert unplayed_episode.item_id == "3003"
        assert unplayed_episode.fully_played is None
        assert unplayed_episode.resume_position_ms is None

    async def test_get_podcast_episodes_raises_for_unknown_podcast(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Get podcast episodes raises MediaNotFoundError for unknown podcast."""
        with pytest.raises(MediaNotFoundError):
            async for _ in provider_with_mocked_data.get_podcast_episodes("9999"):
                pass

    async def test_get_podcast_episode_returns_correct_episode(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Get podcast episode returns correct episode by overcastId."""
        episode = await provider_with_mocked_data.get_podcast_episode("3001")

        assert episode.item_id == "3001"
        assert episode.name == "Episode 1"

    async def test_get_podcast_episode_raises_for_unknown_id(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Get podcast episode raises MediaNotFoundError for unknown ID."""
        with pytest.raises(MediaNotFoundError):
            await provider_with_mocked_data.get_podcast_episode("9999")

    async def test_get_podcast_episode_second_podcast(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Get podcast episode works for episodes in different podcasts."""
        episode = await provider_with_mocked_data.get_podcast_episode("4002")

        assert episode.item_id == "4002"
        assert episode.name == "Episode B"
        assert episode.fully_played is True


class TestRSSIntegration:
    """Tests for RSS enrichment integration in library sync (AC3)."""

    async def test_get_opml_data_enriches_episodes_with_rss(
        self, mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
    ) -> None:
        """
        RSS enrichment populates duration, description, cover_url (AC3.1).

        When get_opml_data() fetches RSS, episodes are enriched with
        description, duration, and cover_url from matching RSS episodes.
        """
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

        # Mock OPML response
        opml_content = """<?xml version='1.0' encoding='UTF-8'?>
<opml version='1.0'>
  <body>
    <outline text='feeds'>
      <outline type='rss' overcastId='2001' title='Test Podcast'
               xmlUrl='https://example.com/feed.xml' htmlUrl='https://example.com'>
        <outline type='podcast-episode' overcastId='3001' title='Episode 1'
                 pubDate='2024-10-12T16:00:00-04:00'
                 enclosureUrl='https://example.com/ep1.mp3'/>
      </outline>
    </outline>
  </body>
</opml>"""

        # Mock RSS response
        rss_dict = {
            "title": "Test Podcast",
            "episodes": [
                {
                    "title": "Episode 1",
                    "description": "Episode 1 description",
                    "total_time": 3600,
                    "episode_art_url": "https://example.com/art1.jpg",
                    "enclosures": [{"url": "https://example.com/ep1.mp3"}],
                }
            ],
            "cover_url": "https://example.com/cover.jpg",
        }

        # Setup mocks
        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value=opml_content)
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        mass_mock.http_session.get = MagicMock(return_value=request_ctx)
        mass_mock.cache.get = AsyncMock(return_value=None)
        mass_mock.cache.set = AsyncMock()

        # Patch get_podcastparser_dict to return our mock RSS
        with patch(
            "music_assistant.providers.overcast.provider.get_podcastparser_dict"
        ) as mock_get_rss:
            mock_get_rss.return_value = rss_dict

            opml_data = await provider._get_opml_data()

            # Verify RSS was fetched for the podcast
            mock_get_rss.assert_called_once()
            call_args = mock_get_rss.call_args
            assert call_args.kwargs["feed_url"] == "https://example.com/feed.xml"

            # Verify episode was enriched
            podcasts = opml_data.get("podcasts", [])
            assert len(podcasts) == 1
            episodes = podcasts[0].get("episodes", [])
            assert len(episodes) == 1
            episode = episodes[0]

            # Check enrichment fields (AC3.1)
            assert episode.get("description") == "Episode 1 description"
            assert episode.get("duration") == 3600
            assert episode.get("cover_url") == "https://example.com/art1.jpg"

            # Verify enriched data was cached (2 calls: enrichment payload + OPML)
            assert mass_mock.cache.set.call_count == 2

            # First call: RSS enrichment payload
            enriched_call = mass_mock.cache.set.call_args_list[0]
            assert enriched_call.kwargs["key"] == CACHE_KEY_RSS_ENRICHMENT
            assert enriched_call.kwargs["data"]["3001"] == {
                "description": "Episode 1 description",
                "duration": 3600,
                "cover_url": "https://example.com/art1.jpg",
            }
            assert enriched_call.kwargs["expiration"] == CACHE_TTL_ENRICHMENT

            # Second call: OPML data with enrichment
            opml_call = mass_mock.cache.set.call_args_list[1]
            assert opml_call.kwargs["key"] == CACHE_KEY_OPML
            cached_episode = opml_call.kwargs["data"]["podcasts"][0]["episodes"][0]
            assert cached_episode.get("description") == "Episode 1 description"

    async def test_get_opml_data_handles_rss_fetch_failure(
        self, mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
    ) -> None:
        """
        Failed RSS fetch falls back to OPML-only data (AC3.3).

        When RSS fetch fails, episodes retain OPML data without enrichment
        (no crash, no data loss).
        """
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

        # Mock OPML response
        opml_content = """<?xml version='1.0' encoding='UTF-8'?>
<opml version='1.0'>
  <body>
    <outline text='feeds'>
      <outline type='rss' overcastId='2001' title='Test Podcast'
               xmlUrl='https://example.com/feed.xml' htmlUrl='https://example.com'>
        <outline type='podcast-episode' overcastId='3001' title='Episode 1'
                 pubDate='2024-10-12T16:00:00-04:00'
                 enclosureUrl='https://example.com/ep1.mp3'/>
      </outline>
    </outline>
  </body>
</opml>"""

        # Setup mocks
        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value=opml_content)
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        mass_mock.http_session.get = MagicMock(return_value=request_ctx)
        mass_mock.cache.get = AsyncMock(return_value=None)
        mass_mock.cache.set = AsyncMock()

        # Patch get_podcastparser_dict to raise exception
        with patch(
            "music_assistant.providers.overcast.provider.get_podcastparser_dict"
        ) as mock_get_rss:
            mock_get_rss.side_effect = Exception("RSS feed unavailable")

            # Should not raise, just log warning and continue
            opml_data = await provider._get_opml_data()

            # Verify podcast and episodes still present
            podcasts = opml_data.get("podcasts", [])
            assert len(podcasts) == 1
            episodes = podcasts[0].get("episodes", [])
            assert len(episodes) == 1
            episode = episodes[0]

            # Episode should have OPML-only data (no enrichment)
            assert episode.get("title") == "Episode 1"
            assert episode.get("enclosure_url") == "https://example.com/ep1.mp3"
            # Enrichment fields should not be present
            assert "description" not in episode
            assert "duration" not in episode

    async def test_get_opml_data_skips_rss_for_already_enriched_episodes(
        self, mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
    ) -> None:
        """
        RSS fetch skipped when all episodes have cached enrichment (AC3.4).

        When the enrichment cache contains metadata for every episode in a podcast,
        the RSS feed should not be fetched for that podcast.
        """
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

        opml_content = """<?xml version='1.0' encoding='UTF-8'?>
<opml version='1.0'>
  <body>
    <outline text='feeds'>
      <outline type='rss' overcastId='2001' title='Test Podcast'
               xmlUrl='https://example.com/feed.xml' htmlUrl='https://example.com'>
        <outline type='podcast-episode' overcastId='3001' title='Episode 1'
                 pubDate='2024-10-12T16:00:00-04:00'
                 enclosureUrl='https://example.com/ep1.mp3'/>
      </outline>
    </outline>
  </body>
</opml>"""

        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value=opml_content)
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        mass_mock.http_session.get = MagicMock(return_value=request_ctx)

        # Cache reads: OPML (pre-lock miss), OPML (re-check under lock, still
        # missing), then the reusable RSS enrichment payload.
        mass_mock.cache.get = AsyncMock(
            side_effect=[
                None,
                None,
                {
                    "3001": {
                        "description": "Cached description",
                        "duration": 3600,
                        "cover_url": "https://example.com/cached.jpg",
                    }
                },
            ]
        )
        mass_mock.cache.set = AsyncMock()

        with patch(
            "music_assistant.providers.overcast.provider.get_podcastparser_dict"
        ) as mock_get_rss:
            opml_data = await provider._get_opml_data()

            # RSS should NOT have been fetched — episode has cached enrichment
            mock_get_rss.assert_not_called()

        episode = opml_data["podcasts"][0]["episodes"][0]
        assert episode["description"] == "Cached description"
        assert episode["duration"] == 3600
        assert episode["cover_url"] == "https://example.com/cached.jpg"

        # Only OPML cache should be set (no enrichment cache update needed)
        assert mass_mock.cache.set.call_count == 1
        assert mass_mock.cache.set.call_args.kwargs["key"] == CACHE_KEY_OPML

    async def test_get_opml_data_fetches_rss_only_for_podcasts_with_new_episodes(
        self, mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
    ) -> None:
        """
        RSS fetch targets only podcasts with unenriched episodes (AC3.5).

        When one podcast has all episodes enriched and another has a new episode,
        only the podcast with the new episode should trigger an RSS fetch.
        """
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

        opml_content = """<?xml version='1.0' encoding='UTF-8'?>
<opml version='1.0'>
  <body>
    <outline text='feeds'>
      <outline type='rss' overcastId='2001' title='Old Podcast'
               xmlUrl='https://example.com/old/feed.xml' htmlUrl='https://example.com/old'>
        <outline type='podcast-episode' overcastId='3001' title='Old Episode'
                 pubDate='2024-10-12T16:00:00-04:00'
                 enclosureUrl='https://example.com/old/ep1.mp3'/>
      </outline>
      <outline type='rss' overcastId='2002' title='New Podcast'
               xmlUrl='https://example.com/new/feed.xml' htmlUrl='https://example.com/new'>
        <outline type='podcast-episode' overcastId='4001' title='New Episode'
                 pubDate='2024-10-12T16:00:00-04:00'
                 enclosureUrl='https://example.com/new/ep1.mp3'/>
      </outline>
    </outline>
  </body>
</opml>"""

        rss_dict = {
            "title": "New Podcast",
            "episodes": [
                {
                    "title": "New Episode",
                    "description": "New description",
                    "total_time": 1800,
                    "enclosures": [{"url": "https://example.com/new/ep1.mp3"}],
                }
            ],
            "cover_url": "https://example.com/new/cover.jpg",
        }

        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value=opml_content)
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        mass_mock.http_session.get = MagicMock(return_value=request_ctx)

        # Episode 3001 has cached enrichment, 4001 is new.
        # Cache reads: OPML (pre-lock miss), OPML (re-check under lock, still
        # missing), then the reusable RSS enrichment payload.
        mass_mock.cache.get = AsyncMock(
            side_effect=[
                None,
                None,
                {"3001": {"description": "Old cached description", "duration": 1200}},
            ]
        )
        mass_mock.cache.set = AsyncMock()

        with patch(
            "music_assistant.providers.overcast.provider.get_podcastparser_dict"
        ) as mock_get_rss:
            mock_get_rss.return_value = rss_dict
            await provider._get_opml_data()

            # Only the new podcast's RSS should be fetched
            mock_get_rss.assert_called_once()
            assert mock_get_rss.call_args.kwargs["feed_url"] == "https://example.com/new/feed.xml"

        # Enrichment cache should keep the old payload and add 4001.
        enriched_call = mass_mock.cache.set.call_args_list[0]
        assert enriched_call.kwargs["key"] == CACHE_KEY_RSS_ENRICHMENT
        assert enriched_call.kwargs["data"]["3001"] == {
            "description": "Old cached description",
            "duration": 1200,
        }
        assert enriched_call.kwargs["data"]["4001"] == {
            "description": "New description",
            "duration": 1800,
            "cover_url": "https://example.com/new/cover.jpg",
        }

    async def test_get_library_podcasts_yields_enriched_data(
        self, mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
    ) -> None:
        """
        Library podcasts with enriched episodes have duration and description (AC3.1).

        When get_library_podcasts() runs with RSS feed available, yielded
        PodcastEpisode objects have enriched metadata.
        """
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

        # Mock OPML response with enriched data
        opml_content = """<?xml version='1.0' encoding='UTF-8'?>
<opml version='1.0'>
  <body>
    <outline text='feeds'>
      <outline type='rss' overcastId='2001' title='Test Podcast'
               xmlUrl='https://example.com/feed.xml' htmlUrl='https://example.com'>
        <outline type='podcast-episode' overcastId='3001' title='Episode 1'
                 pubDate='2024-10-12T16:00:00-04:00'
                 enclosureUrl='https://example.com/ep1.mp3'/>
      </outline>
    </outline>
  </body>
</opml>"""

        # Mock RSS response
        rss_dict = {
            "title": "Test Podcast",
            "episodes": [
                {
                    "title": "Episode 1",
                    "description": "Test episode description",
                    "total_time": 1800,
                    "episode_art_url": "https://example.com/art.jpg",
                    "enclosures": [{"url": "https://example.com/ep1.mp3"}],
                }
            ],
            "cover_url": "https://example.com/cover.jpg",
        }

        # Setup mocks
        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value=opml_content)
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        mass_mock.http_session.get = MagicMock(return_value=request_ctx)
        mass_mock.cache.get = AsyncMock(return_value=None)
        mass_mock.cache.set = AsyncMock()

        # Patch get_podcastparser_dict
        with patch(
            "music_assistant.providers.overcast.provider.get_podcastparser_dict"
        ) as mock_get_rss:
            mock_get_rss.return_value = rss_dict

            # Yield podcasts
            podcasts = []
            async for podcast in provider.get_library_podcasts():
                podcasts.append(podcast)

            assert len(podcasts) == 1
            podcast = podcasts[0]
            assert podcast.name == "Test Podcast"

            # Now get episodes for this podcast
            episodes = []
            async for episode in provider.get_podcast_episodes(podcast.item_id):
                episodes.append(episode)

            assert len(episodes) == 1
            episode = episodes[0]

            # Check enriched metadata (AC3.1)
            assert episode.name == "Episode 1"
            assert episode.duration == 1800
            assert episode.metadata.description == "Test episode description"
            # Check cover image
            assert episode.metadata.images is not None
            assert len(episode.metadata.images) > 0
            assert episode.metadata.images[0].path == "https://example.com/art.jpg"

    async def test_get_library_podcasts_fallback_without_rss(
        self, mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
    ) -> None:
        """
        Library podcasts work without RSS enrichment when fetch fails (AC3.3).

        When RSS fetch fails, get_library_podcasts() still yields Podcast items
        with OPML-only data (no crash, episodes still present).
        """
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

        # Mock OPML response
        opml_content = """<?xml version='1.0' encoding='UTF-8'?>
<opml version='1.0'>
  <body>
    <outline text='feeds'>
      <outline type='rss' overcastId='2001' title='Test Podcast'
               xmlUrl='https://example.com/feed.xml' htmlUrl='https://example.com'>
        <outline type='podcast-episode' overcastId='3001' title='Episode 1'
                 pubDate='2024-10-12T16:00:00-04:00'
                 enclosureUrl='https://example.com/ep1.mp3'/>
      </outline>
    </outline>
  </body>
</opml>"""

        # Setup mocks
        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value=opml_content)
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        mass_mock.http_session.get = MagicMock(return_value=request_ctx)
        mass_mock.cache.get = AsyncMock(return_value=None)
        mass_mock.cache.set = AsyncMock()

        # Patch get_podcastparser_dict to fail
        with patch(
            "music_assistant.providers.overcast.provider.get_podcastparser_dict"
        ) as mock_get_rss:
            mock_get_rss.side_effect = Exception("RSS unavailable")

            # Should still yield podcasts
            podcasts = []
            async for podcast in provider.get_library_podcasts():
                podcasts.append(podcast)

            assert len(podcasts) == 1
            podcast = podcasts[0]
            assert podcast.name == "Test Podcast"
            assert podcast.item_id == "2001"

            # Episodes should still be available
            episodes = []
            async for episode in provider.get_podcast_episodes(podcast.item_id):
                episodes.append(episode)

            assert len(episodes) == 1
            episode = episodes[0]
            assert episode.name == "Episode 1"
            # But no enrichment
            assert episode.duration is None or episode.duration == 0


class TestStreaming:
    """Tests for streaming methods (AC4)."""

    async def test_get_stream_details_returns_stream_details(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Get stream details returns StreamDetails object (AC4.1).

        get_stream_details() should return StreamDetails with correct path,
        stream_type, and content_type for a known episode.
        """
        stream_details = await provider_with_mocked_data.get_stream_details(
            "3001", MediaType.PODCAST_EPISODE
        )

        # Verify StreamDetails object is returned
        assert stream_details is not None
        assert stream_details.path == "https://example.com/ep1.mp3"
        assert stream_details.stream_type == StreamType.HTTP
        assert stream_details.media_type == MediaType.PODCAST_EPISODE
        assert stream_details.provider == "overcast_test"
        assert stream_details.item_id == "3001"

    async def test_get_stream_details_detects_content_type(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Get stream details detects content type from URL (AC4.1).

        Content type should be parsed from enclosure URL extension.
        """
        stream_details = await provider_with_mocked_data.get_stream_details(
            "3001", MediaType.PODCAST_EPISODE
        )

        # Verify content type is detected from .mp3 extension
        assert stream_details.audio_format.content_type == ContentType.MP3

    async def test_get_stream_details_with_auth_token(
        self,
        mass_mock: MagicMock,
        manifest_mock: MagicMock,
        config_mock_with_cookie: MagicMock,
    ) -> None:
        """
        Get stream details passes auth token in URL unchanged (AC4.2).

        Enclosure URL with embedded auth token should be passed through unchanged.
        Uses a standalone provider to avoid mutating the shared fixture.
        """
        auth_url = "https://feeds.example.com/episode.mp3?token=secret_token_123"
        opml_data = {
            "podcasts": [
                {
                    "overcast_id": "9001",
                    "title": "Auth Podcast",
                    "xml_url": "https://example.com/auth.xml",
                    "html_url": "",
                    "episodes": [
                        {
                            "overcast_id": "9002",
                            "title": "Auth Episode",
                            "pub_date": "2024-10-06T16:00:00-04:00",
                            "enclosure_url": auth_url,
                            "overcast_url": "https://overcast.fm/+AUTH",
                            "played": None,
                            "progress": None,
                            "user_deleted": None,
                        },
                    ],
                }
            ],
            "playlists": [],
        }
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)
        provider._cookie = "valid_cookie_value"
        provider._sync_versions = {}
        provider._last_progress_time = {}

        async def mock_get_opml_data() -> dict[str, Any]:
            return opml_data

        provider._get_opml_data = mock_get_opml_data  # type: ignore[method-assign]

        stream_details = await provider.get_stream_details("9002", MediaType.PODCAST_EPISODE)

        assert stream_details.path == auth_url

    async def test_get_stream_details_raises_for_unknown_episode(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Get stream details raises MediaNotFoundError for unknown episode (AC4.3).

        Episode with non-existent overcastId should raise MediaNotFoundError.
        """
        with pytest.raises(MediaNotFoundError):
            await provider_with_mocked_data.get_stream_details("9999", MediaType.PODCAST_EPISODE)

    async def test_get_stream_details_raises_for_episode_without_enclosure(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Get stream details raises MediaNotFoundError for episode without enclosure (AC4.3).

        Episode with no enclosure URL (3004 in fixture) should raise MediaNotFoundError.
        """
        with pytest.raises(MediaNotFoundError):
            await provider_with_mocked_data.get_stream_details("3004", MediaType.PODCAST_EPISODE)

    async def test_get_stream_details_second_podcast_episode(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Get stream details works for episodes in different podcasts.

        Should find episodes across all podcasts in OPML data.
        """
        stream_details = await provider_with_mocked_data.get_stream_details(
            "4002", MediaType.PODCAST_EPISODE
        )

        assert stream_details.path == "https://example.com/epB.mp3"
        assert stream_details.item_id == "4002"


class TestPlaybackProgressWriteBack:
    """Tests for on_played() and progress write-back (AC5)."""

    async def test_on_played_ignores_non_podcast_media_types(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        on_played() returns early for non-podcast media types.

        Should not POST to set_progress for TRACK, ALBUM, etc.
        """
        provider = provider_with_mocked_data

        # Call with TRACK media type - should return early
        await provider.on_played(
            media_type=MediaType.TRACK,
            prov_item_id="3001",
            fully_played=False,
            position=30,
            media_item=None,  # type: ignore[arg-type]
            is_playing=True,
        )

        # Verify no HTTP POST was made
        provider.mass.http_session.post.assert_not_called()  # type: ignore[attr-defined]

    async def test_on_played_ignores_none_media_item(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        on_played() returns early when media_item is None.

        Should not POST to set_progress without media item.
        """
        provider = provider_with_mocked_data

        # Call with None media_item - should return early
        await provider.on_played(
            media_type=MediaType.PODCAST_EPISODE,
            prov_item_id="3001",
            fully_played=False,
            position=30,
            media_item=None,  # type: ignore[arg-type]
            is_playing=True,
        )

        # Verify no HTTP POST was made
        provider.mass.http_session.post.assert_not_called()  # type: ignore[attr-defined]

    async def test_on_played_ignores_wrong_media_item_type(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        on_played() returns early when media_item is not PodcastEpisode.

        Should not POST to set_progress for non-PodcastEpisode items.
        """
        provider = provider_with_mocked_data

        # Create a mock media_item that is NOT a PodcastEpisode
        media_item = MagicMock()
        media_item.__class__.__name__ = "Track"

        # Call with non-PodcastEpisode - should return early
        await provider.on_played(
            media_type=MediaType.PODCAST_EPISODE,
            prov_item_id="3001",
            fully_played=False,
            position=30,
            media_item=media_item,
            is_playing=True,
        )

        # Verify no HTTP POST was made
        provider.mass.http_session.post.assert_not_called()  # type: ignore[attr-defined]

    async def test_on_played_posts_progress_during_playback(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        on_played() with is_playing=True POSTs progress (AC5.1).

        When is_playing=True, should POST progress after guard interval.
        """
        provider = provider_with_mocked_data
        # Create a mock PodcastEpisode (isinstance check will pass for it)
        episode = MagicMock(spec=PodcastEpisode)

        # Mock POST response
        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value="42")
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        # Call on_played with is_playing=True
        await provider.on_played(
            media_type=MediaType.PODCAST_EPISODE,
            prov_item_id="3001",
            fully_played=False,
            position=120,
            media_item=episode,
            is_playing=True,
        )

        # Verify POST was made
        provider.mass.http_session.post.assert_called_once()
        call_args = provider.mass.http_session.post.call_args
        assert "3001" in call_args[0][0]  # URL contains episode ID
        assert call_args.kwargs["data"]["p"] == "120"  # Position in seconds
        assert call_args.kwargs["data"]["v"] == "0"  # First POST uses v=0

    async def test_on_played_respects_guard_interval_during_playback(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        on_played() during playback only POSTs once per 60 seconds (AC5.1).

        Second call within 60 seconds should be skipped.
        """
        provider = provider_with_mocked_data
        episode = MagicMock(spec=PodcastEpisode)

        # Mock POST response
        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value="0")
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        with patch("time.time") as mock_time:
            # First call at time 1000
            mock_time.return_value = 1000
            await provider.on_played(
                media_type=MediaType.PODCAST_EPISODE,
                prov_item_id="3001",
                fully_played=False,
                position=120,
                media_item=episode,
                is_playing=True,
            )

            # Verify first POST happened
            assert provider.mass.http_session.post.call_count == 1

            # Second call at time within guard window
            # Use 1000 + (GUARD_INTERVAL_PLAYING - 1) to be just inside the guard window
            mock_time.return_value = 1000 + GUARD_INTERVAL_PLAYING - 1
            await provider.on_played(
                media_type=MediaType.PODCAST_EPISODE,
                prov_item_id="3001",
                fully_played=False,
                position=150,
                media_item=episode,
                is_playing=True,
            )

            # Verify second POST was skipped (still 1 call)
            assert provider.mass.http_session.post.call_count == 1

    async def test_on_played_posts_immediately_on_pause(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        on_played() with is_playing=False POSTs within 1 second (AC5.2).

        When is_playing=False (pause/stop), should POST promptly.
        """
        provider = provider_with_mocked_data
        episode = MagicMock(spec=PodcastEpisode)

        # Mock POST response
        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value="0")
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        with patch("time.time") as mock_time:
            # First playback POST at time 1000
            mock_time.return_value = 1000
            await provider.on_played(
                media_type=MediaType.PODCAST_EPISODE,
                prov_item_id="3001",
                fully_played=False,
                position=120,
                media_item=episode,
                is_playing=True,
            )
            assert provider.mass.http_session.post.call_count == 1

            # Pause at time 1000 + GUARD_INTERVAL_PLAYING (well past playback guard)
            # Pause should POST immediately because its guard is only GUARD_INTERVAL_ACTION
            mock_time.return_value = 1000 + GUARD_INTERVAL_PLAYING
            await provider.on_played(
                media_type=MediaType.PODCAST_EPISODE,
                prov_item_id="3001",
                fully_played=False,
                position=150,
                media_item=episode,
                is_playing=False,
            )

            # Verify pause POST happened (now 2 calls)
            assert provider.mass.http_session.post.call_count == 2

    async def test_on_played_sends_finished_sentinel_on_completion(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        on_played() with fully_played=True sends FINISHED_SENTINEL (AC5.3).

        When fully_played=True, should POST p=2147483647.
        """
        provider = provider_with_mocked_data
        episode = MagicMock(spec=PodcastEpisode)

        # Mock POST response
        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value="0")
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        # Call with fully_played=True
        await provider.on_played(
            media_type=MediaType.PODCAST_EPISODE,
            prov_item_id="3001",
            fully_played=True,
            position=3600,
            media_item=episode,
            is_playing=False,
        )

        # Verify POST with FINISHED_SENTINEL
        provider.mass.http_session.post.assert_called_once()
        call_args = provider.mass.http_session.post.call_args
        assert call_args.kwargs["data"]["p"] == str(FINISHED_SENTINEL)

    async def test_on_played_completion_bypasses_guard_after_heartbeat(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Completion should not be skipped when it follows a recent heartbeat."""
        provider = provider_with_mocked_data
        episode = MagicMock(spec=PodcastEpisode)

        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value="0")
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        with patch("time.time") as mock_time:
            mock_time.return_value = 1000
            await provider.on_played(
                media_type=MediaType.PODCAST_EPISODE,
                prov_item_id="3001",
                fully_played=False,
                position=120,
                media_item=episode,
                is_playing=True,
            )

            mock_time.return_value = 1000.5
            await provider.on_played(
                media_type=MediaType.PODCAST_EPISODE,
                prov_item_id="3001",
                fully_played=True,
                position=3600,
                media_item=episode,
                is_playing=False,
            )

        assert provider.mass.http_session.post.call_count == 2
        second_call_args = provider.mass.http_session.post.call_args_list[1]
        assert second_call_args.kwargs["data"]["p"] == str(FINISHED_SENTINEL)

    async def test_on_played_mark_unplayed_bypasses_guard_after_heartbeat(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Mark-unplayed should not be skipped when it follows a recent heartbeat."""
        provider = provider_with_mocked_data
        episode = MagicMock(spec=PodcastEpisode)

        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value="0")
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        with patch("time.time") as mock_time:
            mock_time.return_value = 1000
            await provider.on_played(
                media_type=MediaType.PODCAST_EPISODE,
                prov_item_id="3001",
                fully_played=False,
                position=120,
                media_item=episode,
                is_playing=True,
            )

            mock_time.return_value = 1000.5
            await provider.on_played(
                media_type=MediaType.PODCAST_EPISODE,
                prov_item_id="3001",
                fully_played=False,
                position=0,
                media_item=episode,
                is_playing=False,
            )

        assert provider.mass.http_session.post.call_count == 2
        second_call_args = provider.mass.http_session.post.call_args_list[1]
        assert second_call_args.kwargs["data"]["p"] == "0"

    async def test_on_played_first_post_uses_sync_version_zero(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        First POST uses v=0, subsequent POSTs use version from response (AC5.4).

        First POST to episode should use v=0 initial sync version.
        """
        provider = provider_with_mocked_data
        episode = MagicMock(spec=PodcastEpisode)

        # Mock POST response
        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value="42")
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        # First POST - should use v=0
        await provider.on_played(
            media_type=MediaType.PODCAST_EPISODE,
            prov_item_id="3001",
            fully_played=False,
            position=120,
            media_item=episode,
            is_playing=False,
        )

        call_args = provider.mass.http_session.post.call_args
        assert call_args.kwargs["data"]["v"] == "0"

    async def test_on_played_subsequent_post_uses_sync_version_from_response(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Second POST uses sync version from first response (AC5.4).

        After first POST returns v=42, second POST should use v=42.
        """
        provider = provider_with_mocked_data
        episode = MagicMock(spec=PodcastEpisode)

        # Mock POST response
        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value="42")
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        with patch("time.time") as mock_time:
            # First POST at time 1000
            mock_time.return_value = 1000
            await provider.on_played(
                media_type=MediaType.PODCAST_EPISODE,
                prov_item_id="3001",
                fully_played=False,
                position=120,
                media_item=episode,
                is_playing=False,
            )

            # Second POST at time 1000 + GUARD_INTERVAL_ACTION (after action guard passes)
            mock_time.return_value = 1000 + GUARD_INTERVAL_ACTION
            await provider.on_played(
                media_type=MediaType.PODCAST_EPISODE,
                prov_item_id="3001",
                fully_played=False,
                position=150,
                media_item=episode,
                is_playing=False,
            )

            # Verify second POST used v=42
            assert provider.mass.http_session.post.call_count == 2
            second_call_args = provider.mass.http_session.post.call_args_list[1]
            assert second_call_args.kwargs["data"]["v"] == "42"

    async def test_on_played_only_calls_set_progress_endpoint(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        on_played() only calls set_progress, not separate save endpoint (AC5.5).

        Verify exactly one HTTP POST per progress update (no separate save).
        """
        provider = provider_with_mocked_data
        episode = MagicMock(spec=PodcastEpisode)

        # Mock POST response
        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value="0")
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        # Call on_played
        await provider.on_played(
            media_type=MediaType.PODCAST_EPISODE,
            prov_item_id="3001",
            fully_played=False,
            position=120,
            media_item=episode,
            is_playing=False,
        )

        # Verify exactly 1 POST was made
        assert provider.mass.http_session.post.call_count == 1

    async def test_on_played_catches_post_errors(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        on_played() handles POST errors gracefully without interrupting playback.

        Errors should be logged but not raised.
        """
        provider = provider_with_mocked_data
        episode = MagicMock(spec=PodcastEpisode)

        # Mock POST to raise exception
        provider.mass.http_session.post = AsyncMock(side_effect=Exception("Network error"))  # type: ignore[method-assign]

        # Should not raise, just log warning
        await provider.on_played(
            media_type=MediaType.PODCAST_EPISODE,
            prov_item_id="3001",
            fully_played=False,
            position=120,
            media_item=episode,
            is_playing=False,
        )

        # Verify POST was attempted
        provider.mass.http_session.post.assert_called_once()


class TestGetResumePosition:
    """Tests for get_resume_position() method."""

    @pytest.fixture
    async def provider_for_resume(
        self, mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
    ) -> OvercastProvider:
        """Provide a provider instance with mocked OPML data."""
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

        # Initialize per-episode state tracking dicts (normally done in handle_async_init)
        provider._cookie = "test_cookie"
        provider._sync_versions = {}
        provider._last_progress_time = {}

        # Mock OPML data
        opml_data = {
            "podcasts": [
                {
                    "overcast_id": "2001",
                    "title": "Test Podcast",
                    "episodes": [
                        {
                            "overcast_id": "1001",
                            "overcast_url": "https://overcast.fm/+test1",
                            "title": "Test Episode",
                        }
                    ],
                }
            ],
            "playlists": [],
        }

        # Mock the _get_opml_data method
        provider._get_opml_data = AsyncMock(return_value=opml_data)  # type: ignore[method-assign]

        return provider

    async def test_get_resume_position_returns_start_time_as_milliseconds(
        self, provider_for_resume: OvercastProvider
    ) -> None:
        """
        get_resume_position() fetches episode page and returns resume time (AC6.1).

        Episode page contains data-start-time="120", provider returns (False, 120000).
        """
        provider = provider_for_resume

        # Mock episode page HTML
        html_content = '<audio id="audioplayer" data-start-time="120" data-sync-version="5">'

        # Mock HTTP GET response with proper async context manager
        response = AsyncMock()
        response.text = AsyncMock(return_value=html_content)
        response.raise_for_status = MagicMock()

        # Create a mock context manager for the HTTP request
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.get = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        # Call get_resume_position
        fully_played, resume_position_ms, timestamp = await provider.get_resume_position(
            item_id="1001", media_type=MediaType.PODCAST_EPISODE
        )

        # Verify results
        assert fully_played is False
        assert resume_position_ms == 120000  # 120 seconds converted to milliseconds
        assert timestamp is None

    async def test_get_resume_position_caches_sync_version(
        self, provider_for_resume: OvercastProvider
    ) -> None:
        """
        get_resume_position() caches sync_version for write-back (AC6.2).

        Episode page contains data-sync-version="42", verify it's cached in provider._sync_versions.
        """
        provider = provider_for_resume

        # Mock episode page HTML with sync version
        html_content = '<audio id="audioplayer" data-start-time="300" data-sync-version="42">'

        # Mock HTTP GET response with proper async context manager
        response = AsyncMock()
        response.text = AsyncMock(return_value=html_content)
        response.raise_for_status = MagicMock()

        # Create a mock context manager for the HTTP request
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.get = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        # Call get_resume_position
        _fully_played, _resume_position_ms, _timestamp = await provider.get_resume_position(
            item_id="1001", media_type=MediaType.PODCAST_EPISODE
        )

        # Verify sync version is cached
        assert provider._sync_versions.get("1001") == 42

    async def test_get_resume_position_returns_fully_played_for_sentinel(
        self, provider_for_resume: OvercastProvider
    ) -> None:
        """
        get_resume_position() returns fully_played=True when start_time >= FINISHED_SENTINEL.

        Episode page contains data-start-time >= FINISHED_SENTINEL, verify returns (True, 0).
        """
        provider = provider_for_resume

        # Mock episode page HTML with sentinel value
        sentinel_value = FINISHED_SENTINEL
        html_content = (
            f'<audio id="audioplayer" data-start-time="{sentinel_value}" data-sync-version="5">'
        )

        # Mock HTTP GET response with proper async context manager
        response = AsyncMock()
        response.text = AsyncMock(return_value=html_content)
        response.raise_for_status = MagicMock()

        # Create a mock context manager for the HTTP request
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.get = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        # Call get_resume_position
        fully_played, resume_position_ms, timestamp = await provider.get_resume_position(
            item_id="1001", media_type=MediaType.PODCAST_EPISODE
        )

        # Verify fully_played is True
        assert fully_played is True
        assert resume_position_ms == 0
        assert timestamp is None

    async def test_get_resume_position_falls_back_on_fetch_failure(
        self, provider_for_resume: OvercastProvider
    ) -> None:
        """
        get_resume_position() raises NotImplementedError on fetch failure (AC6.3).

        HTTP GET fails, provider raises NotImplementedError for MA to fall back to playlog.
        """
        provider = provider_for_resume

        # Mock HTTP GET to raise exception
        provider.mass.http_session.get = MagicMock(  # type: ignore[method-assign]
            side_effect=aiohttp.ClientError("Connection failed")
        )

        # Verify NotImplementedError is raised
        with pytest.raises(NotImplementedError):
            await provider.get_resume_position(item_id="1001", media_type=MediaType.PODCAST_EPISODE)

    async def test_get_resume_position_falls_back_on_parse_failure(
        self, provider_for_resume: OvercastProvider
    ) -> None:
        """
        get_resume_position() raises NotImplementedError when parsing fails.

        Episode page HTML is invalid or lacks audio player, provider raises NotImplementedError.
        """
        provider = provider_for_resume

        # Mock episode page HTML without audio player
        html_content = "<div>No audio player here</div>"

        # Mock HTTP GET response with proper async context manager
        response = AsyncMock()
        response.text = AsyncMock(return_value=html_content)
        response.raise_for_status = MagicMock()

        # Create a mock context manager for the HTTP request
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.get = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        # Verify NotImplementedError is raised
        with pytest.raises(NotImplementedError):
            await provider.get_resume_position(item_id="1001", media_type=MediaType.PODCAST_EPISODE)


# Browse tests (AC7)
class TestBrowse:
    """Test browse() method for Overcast provider."""

    async def test_browse_root_returns_playlists_and_podcasts(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        browse() at root returns Playlists and Podcasts folders (AC7.1).

        Root browse path shows exactly 2 BrowseFolder items for Playlists and Podcasts.
        """
        provider = provider_with_mocked_data
        base = f"{provider.instance_id}://"

        result = await provider.browse(base)

        assert len(result) == 2
        item_names = {item.name for item in result}
        assert "Playlists" in item_names
        assert "Podcasts" in item_names

        # Verify paths are correct (root listing is BrowseFolder-only)
        for item in result:
            assert isinstance(item, BrowseFolder)
            if item.name == "Playlists":
                assert item.path == f"{base}{BROWSE_PLAYLISTS}"
            elif item.name == "Podcasts":
                assert item.path == f"{base}{BROWSE_PODCASTS}"

    async def test_browse_playlists_folder_lists_all_playlists(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        browse() playlists folder lists all playlists (AC7.2).

        Playlists folder shows BrowseFolder for each playlist in test data,
        with smart playlists labeled.
        """
        provider = provider_with_mocked_data
        base = f"{provider.instance_id}://"
        path = f"{base}{BROWSE_PLAYLISTS}"

        result = await provider.browse(path)

        assert len(result) == 2
        names = {item.name for item in result}
        assert "All Episodes (Smart)" in names
        assert "Queue" in names

    async def test_browse_playlist_episodes_returns_podcast_episodes(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        browse() playlist episodes returns PodcastEpisode items (AC7.3).

        Navigates into a playlist using the path from the listing (not a
        hardcoded string) to verify the path roundtrip works end-to-end.
        """
        provider = provider_with_mocked_data
        base = f"{provider.instance_id}://"

        # First, get the playlist listing to obtain the real path
        playlists = await provider.browse(f"{base}{BROWSE_PLAYLISTS}")
        queue_folder = next(item for item in playlists if item.name == "Queue")
        assert isinstance(queue_folder, BrowseFolder)

        # Navigate into the playlist using its actual path
        result = await provider.browse(queue_folder.path)

        # Queue playlist contains episodes 3002 and 4001
        assert len(result) == 2
        episode_ids = {item.item_id for item in result}
        assert "3002" in episode_ids
        assert "4001" in episode_ids

    async def test_browse_playlist_with_spaces_in_title(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        browse() handles playlist titles with spaces correctly (AC7.3).

        Playlist titles containing spaces must survive the path encoding
        roundtrip (listing → path → navigation → title lookup).
        """
        provider = provider_with_mocked_data
        base = f"{provider.instance_id}://"

        # Get the playlist listing
        playlists = await provider.browse(f"{base}{BROWSE_PLAYLISTS}")
        all_episodes_folder = next(item for item in playlists if "All Episodes" in item.name)
        assert isinstance(all_episodes_folder, BrowseFolder)

        # Navigate using the path from the listing
        result = await provider.browse(all_episodes_folder.path)

        # "All Episodes" playlist contains episodes 3001-3003 + 4001-4002 (all have enclosures)
        assert len(result) == 5

    async def test_browse_podcasts_folder_lists_all_podcasts(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        browse() podcasts folder lists all podcasts (AC7.4).

        Podcasts folder shows BrowseFolder for each podcast in test data.
        """
        provider = provider_with_mocked_data
        base = f"{provider.instance_id}://"
        path = f"{base}{BROWSE_PODCASTS}"

        result = await provider.browse(path)

        assert len(result) == 2
        names = {item.name for item in result}
        assert "Podcast One" in names
        assert "Podcast Two" in names

    async def test_browse_podcast_episodes_returns_episodes_and_unplayed_folder(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        browse() podcast episodes returns episodes plus Unplayed folder (AC7.5).

        Clicking a podcast shows its episodes as PodcastEpisode items
        with an "Unplayed" BrowseFolder at the start.
        """
        provider = provider_with_mocked_data
        base = f"{provider.instance_id}://"
        path = f"{base}{BROWSE_PODCASTS}/2001"

        result = await provider.browse(path)

        # Should have 1 folder (Unplayed) + 3 episodes with enclosure URLs
        # (episode 3004 has no enclosure, so it's excluded)
        assert len(result) == 4

        # First item must be the Unplayed BrowseFolder, not an episode
        assert isinstance(result[0], BrowseFolder)
        assert result[0].name == "Unplayed"

        # Remaining items must all be PodcastEpisode instances
        for item in result[1:]:
            assert isinstance(item, PodcastEpisode)
        episode_ids = {item.item_id for item in result[1:]}
        assert episode_ids == {"3001", "3002", "3003"}

    async def test_browse_unplayed_episodes_filters_played(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        browse() unplayed folder returns only unplayed episodes (AC7.6).

        Unplayed folder shows only episodes where played != "1".
        """
        provider = provider_with_mocked_data
        base = f"{provider.instance_id}://"
        path = f"{base}{BROWSE_PODCASTS}/2001/{BROWSE_UNPLAYED}"

        result = await provider.browse(path)

        # Podcast One: 3001 is played, 3002 is progress (unplayed), 3003 is unplayed
        # Should only get 3002 and 3003, no BrowseFolder
        assert len(result) == 2
        for item in result:
            assert isinstance(item, PodcastEpisode)
        episode_ids = {item.item_id for item in result}
        assert episode_ids == {"3002", "3003"}

    async def test_browse_unplayed_episodes_podcast_two(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        browse() unplayed folder for Podcast Two returns correct episodes (AC7.6).

        Podcast Two: 4001 is unplayed, 4002 is played.
        Should only return 4001.
        """
        provider = provider_with_mocked_data
        base = f"{provider.instance_id}://"
        path = f"{base}{BROWSE_PODCASTS}/2002/{BROWSE_UNPLAYED}"

        result = await provider.browse(path)

        # Should only get 4001
        assert len(result) == 1
        assert result[0].item_id == "4001"
        assert "4002" not in {item.item_id for item in result}


# Error Handling tests (AC8)
class TestErrorHandling:
    """Test error handling and resilience (AC8)."""

    async def test_fetch_opml_429_raises_rate_limited(
        self, mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
    ) -> None:
        """OPML 429 response is converted to the shared retryable rate-limit error."""
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)
        provider.throttler = ThrottlerManager(rate_limit=1, period=1, retry_attempts=1)

        response = AsyncMock()
        response.status = 429
        response.headers = {"Retry-After": "42"}

        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        mass_mock.http_session.get = MagicMock(return_value=request_ctx)

        with pytest.raises(RetriesExhausted):
            await provider._fetch_opml()

        assert mass_mock.http_session.get.call_count == 1

    async def test_fetch_opml_network_error_is_retryable(
        self, mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
    ) -> None:
        """Network errors are converted to retryable temporary failures."""
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)
        provider.throttler = ThrottlerManager(rate_limit=1, period=1, retry_attempts=1)

        request_ctx = AsyncMock()
        request_ctx.__aenter__.side_effect = aiohttp.ClientError("Connection failed")
        mass_mock.http_session.get = MagicMock(return_value=request_ctx)

        with pytest.raises(RetriesExhausted):
            await provider._fetch_opml()

        assert mass_mock.http_session.get.call_count == 1

    async def test_fetch_opml_401_raises_login_failed(
        self, mass_mock: MagicMock, manifest_mock: MagicMock, config_mock_with_cookie: MagicMock
    ) -> None:
        """
        OPML 401 response raises LoginFailed (AC8.3).

        When Overcast returns 401, _fetch_opml() should raise LoginFailed
        with message about expired cookie.
        """
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

        # Mock 401 response
        response = AsyncMock()
        response.status = 401

        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        mass_mock.http_session.get = MagicMock(return_value=request_ctx)

        # Should raise LoginFailed
        with pytest.raises(LoginFailed) as exc_info:
            await provider._fetch_opml()

        assert "expired" in str(exc_info.value).lower()

    async def test_get_opml_data_falls_back_to_cache_on_429(
        self,
        mass_mock: MagicMock,
        manifest_mock: MagicMock,
        config_mock_with_cookie: MagicMock,
        sample_opml_data: dict[str, Any],
    ) -> None:
        """
        OPML fetch 429 causes fallback to cached data (AC8.1).

        When cache returns data on first call, _get_opml_data() should return
        the cached data (earliest fallback).
        """
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

        # Mock cache to return sample data on first call
        cached_data = sample_opml_data
        mass_mock.cache.get = AsyncMock(return_value=cached_data)

        # Call _get_opml_data
        result = await provider._get_opml_data()

        # Should return cached data
        assert result == cached_data

    async def test_get_opml_data_raises_when_cache_empty_and_fetch_fails(
        self,
        mass_mock: MagicMock,
        manifest_mock: MagicMock,
        config_mock_with_cookie: MagicMock,
    ) -> None:
        """OPML fetch raises a user-facing error when retries exhaust and cache is empty."""
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

        mass_mock.cache.get = AsyncMock(return_value=None)

        async def mock_fetch_opml() -> str:
            raise RetriesExhausted("Retries exhausted")

        provider._fetch_opml = mock_fetch_opml  # type: ignore[method-assign]

        with pytest.raises(MusicAssistantError):
            await provider._get_opml_data()

    async def test_post_progress_client_error_logs_warning(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        set_progress POST failure logs warning (AC8.2).

        When POST to set_progress fails (e.g., network error), should log
        warning but not raise exception (playback continues).
        """
        provider = provider_with_mocked_data

        # Mock HTTP session to raise error
        provider.mass.http_session.post = AsyncMock(  # type: ignore[method-assign]
            side_effect=aiohttp.ClientError("Network error")
        )

        # Call _post_progress - should not raise
        await provider._post_progress("3001", 120)

        # Verify last_progress_time was NOT updated (failure case)
        assert (
            "3001" not in provider._last_progress_time or provider._last_progress_time["3001"] == 0
        )

    async def test_post_progress_401_raises_login_failed(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        set_progress POST 401 raises LoginFailed (AC8.2).

        When POST returns 401, _post_progress should raise LoginFailed
        so the caller (on_played) can propagate it to the framework.
        """
        provider = provider_with_mocked_data

        # Mock 401 response
        response = AsyncMock()
        response.status = 401

        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        # Call _post_progress - should raise LoginFailed
        with pytest.raises(LoginFailed):
            await provider._post_progress("3001", 120)

    async def test_on_played_exception_doesnt_interrupt_playback(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        on_played() exception doesn't interrupt playback (AC8.2).

        When _post_progress fails, on_played() should catch the exception
        and log warning, never re-raising.
        """
        provider = provider_with_mocked_data

        # Create a minimal episode object for testing
        # (use mock to avoid complex construction)
        episode = MagicMock(spec=PodcastEpisode)
        episode.item_id = "3001"

        # Mock _post_progress to raise an exception
        provider._post_progress = AsyncMock(side_effect=Exception("Unexpected error"))  # type: ignore[method-assign]

        # Call on_played - should NOT raise
        await provider.on_played(
            media_type=MediaType.PODCAST_EPISODE,
            prov_item_id="3001",
            fully_played=False,
            position=120,
            media_item=episode,
            is_playing=True,
        )

        # Verify _post_progress was called
        provider._post_progress.assert_called_once()

    async def test_on_played_login_failed_propagates(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        on_played() propagates LoginFailed to the framework (AC8.2).

        When _post_progress raises LoginFailed (expired cookie),
        on_played() must re-raise so the framework prompts re-authentication.
        """
        provider = provider_with_mocked_data

        episode = MagicMock(spec=PodcastEpisode)
        episode.item_id = "3001"

        provider._post_progress = AsyncMock(side_effect=LoginFailed("Cookie expired"))  # type: ignore[method-assign]

        with pytest.raises(LoginFailed):
            await provider.on_played(
                media_type=MediaType.PODCAST_EPISODE,
                prov_item_id="3001",
                fully_played=False,
                position=120,
                media_item=episode,
                is_playing=True,
            )

    async def test_post_progress_server_error_updates_timestamp(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        5xx response updates _last_progress_time so the guard limits retries.

        Without this, an Overcast outage would let every on_played call bypass
        the guard and hammer a broken endpoint at heartbeat rate.
        """
        provider = provider_with_mocked_data

        response = AsyncMock()
        response.status = 503
        response.text = AsyncMock(return_value="upstream broken")
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        with patch("time.time") as mock_time:
            mock_time.return_value = 9999.0
            await provider._post_progress("3001", 120)

        assert provider._last_progress_time.get("3001") == 9999.0

    async def test_post_progress_4xx_non_auth_updates_timestamp(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Non-auth 4xx (e.g. 400 bad request) also updates the rate-limit timestamp."""
        provider = provider_with_mocked_data

        response = AsyncMock()
        response.status = 400
        response.text = AsyncMock(return_value="bad request")
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        provider.mass.http_session.post = MagicMock(return_value=request_ctx)  # type: ignore[method-assign]

        with patch("time.time") as mock_time:
            mock_time.return_value = 7777.0
            await provider._post_progress("3001", 120)

        assert provider._last_progress_time.get("3001") == 7777.0

    async def test_post_progress_rejects_non_numeric_episode_id(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Non-numeric episode_id is refused before any URL is constructed.

        overcastId comes from external OPML XML — defense-in-depth: never
        let an attacker-influenced value flow into the request URL even
        though aiohttp would ultimately reject path-traversal attempts.
        """
        provider = provider_with_mocked_data
        provider.mass.http_session.post = MagicMock()  # type: ignore[method-assign]

        for bad_id in ("../../etc/passwd", "abc", "1234?injected=1", "12 34", ""):
            await provider._post_progress(bad_id, 120)
            provider.mass.http_session.post.assert_not_called()

    async def test_get_opml_data_propagates_login_failed_when_cache_empty(
        self,
        mass_mock: MagicMock,
        manifest_mock: MagicMock,
        config_mock_with_cookie: MagicMock,
    ) -> None:
        """LoginFailed from _fetch_opml propagates instead of being swallowed."""
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)
        mass_mock.cache.get = AsyncMock(return_value=None)

        async def mock_fetch_opml() -> str:
            raise LoginFailed("Cookie expired")

        provider._fetch_opml = mock_fetch_opml  # type: ignore[method-assign]

        with pytest.raises(LoginFailed):
            await provider._get_opml_data()

    async def test_get_opml_data_rss_failure_falls_back_to_opml_only(
        self,
        mass_mock: MagicMock,
        manifest_mock: MagicMock,
        config_mock_with_cookie: MagicMock,
    ) -> None:
        """
        RSS enrichment failures never abort the OPML sync.

        RSS feeds are fetched unauthenticated via the shared session, so
        ``LoginFailed`` cannot originate there in practice — but any error
        raised by the RSS fetch (including a stray ``LoginFailed``) must be
        swallowed and degrade gracefully to OPML-only metadata rather than
        propagating and failing the whole library sync.
        """
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)

        opml_content = """<?xml version='1.0' encoding='UTF-8'?>
<opml version='1.0'>
  <body>
    <outline text='feeds'>
      <outline type='rss' overcastId='2001' title='Test Podcast'
               xmlUrl='https://example.com/feed.xml' htmlUrl='https://example.com'>
        <outline type='podcast-episode' overcastId='3001' title='Episode 1'
                 pubDate='2024-10-12T16:00:00-04:00'
                 enclosureUrl='https://example.com/ep1.mp3'/>
      </outline>
    </outline>
  </body>
</opml>"""

        response = AsyncMock()
        response.status = 200
        response.text = AsyncMock(return_value=opml_content)
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        mass_mock.http_session.get = MagicMock(return_value=request_ctx)
        mass_mock.cache.get = AsyncMock(return_value=None)
        mass_mock.cache.set = AsyncMock()

        with patch(
            "music_assistant.providers.overcast.provider.get_podcastparser_dict"
        ) as mock_get_rss:
            mock_get_rss.side_effect = LoginFailed("Cookie expired")

            # Must NOT raise — the failure degrades to OPML-only data.
            result = await provider._get_opml_data()

        # OPML-only data is still returned with the podcast intact.
        assert len(result["podcasts"]) == 1
        assert result["podcasts"][0]["overcast_id"] == "2001"
        # OPML was cached despite the RSS failure.
        mass_mock.cache.set.assert_any_call(
            key=CACHE_KEY_OPML,
            provider=provider.instance_id,
            category=CACHE_CATEGORY_OPML,
            data=result,
            expiration=CACHE_TTL,
        )

    async def test_fetch_opml_unexpected_redirect_is_retryable(
        self,
        mass_mock: MagicMock,
        manifest_mock: MagicMock,
        config_mock_with_cookie: MagicMock,
    ) -> None:
        """Non-/login 3xx responses become retryable temporary failures."""
        provider = OvercastProvider(mass_mock, manifest_mock, config_mock_with_cookie)
        provider.throttler = ThrottlerManager(rate_limit=1, period=1, retry_attempts=1)

        response = AsyncMock()
        response.status = 302
        response.headers = {"Location": "https://overcast.fm/account/"}
        request_ctx = AsyncMock()
        request_ctx.__aenter__.return_value = response
        request_ctx.__aexit__.return_value = None

        mass_mock.http_session.get = MagicMock(return_value=request_ctx)

        with pytest.raises(RetriesExhausted):
            await provider._fetch_opml()

        assert mass_mock.http_session.get.call_count == 1


class TestBrowseEdgeCases:
    """Tests for browse() handling of malformed or unknown paths."""

    async def test_browse_invalid_base_returns_empty(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Path that doesn't start with the instance base returns []."""
        result = await provider_with_mocked_data.browse("notovercast://playlists")
        assert result == []

    async def test_browse_unknown_top_level_returns_empty(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """Unknown top-level subpath returns []."""
        provider = provider_with_mocked_data
        result = await provider.browse(f"{provider.instance_id}://nonsense")
        assert result == []

    async def test_browse_unknown_podcast_id_returns_only_unplayed_folder(
        self, provider_with_mocked_data: OvercastProvider
    ) -> None:
        """
        Unknown podcast ID returns only the Unplayed folder, no episodes.

        The browse layer always emits the Unplayed folder shortcut; an
        unknown podcast id simply has no episodes to list under it.
        """
        provider = provider_with_mocked_data
        result = await provider.browse(f"{provider.instance_id}://podcasts/9999")
        # Only the Unplayed folder, no episodes from a non-existent podcast
        assert len(result) == 1
        assert isinstance(result[0], BrowseFolder)
        assert result[0].name == "Unplayed"
