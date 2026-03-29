"""Test OPML parsing helpers for Overcast provider."""

from pathlib import Path
from typing import Any

import pytest
from music_assistant_models.enums import ContentType

from music_assistant.providers.overcast.parsing import (
    enrich_episodes_from_rss,
    opml_episode_to_mass_episode,
    opml_podcast_to_mass_podcast,
    parse_episode_page,
    parse_opml,
)


@pytest.fixture
def sample_opml_content() -> str:
    """Load sample OPML XML fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "opml_sample.xml"
    return fixture_path.read_text()


class TestParseOpml:
    """Tests for parse_opml() function."""

    def test_parse_opml_returns_dict_with_podcasts_and_playlists(
        self, sample_opml_content: str
    ) -> None:
        """Parse OPML returns dict with podcasts and playlists keys."""
        result = parse_opml(sample_opml_content)

        assert isinstance(result, dict)
        assert "podcasts" in result
        assert "playlists" in result
        assert isinstance(result["podcasts"], list)
        assert isinstance(result["playlists"], list)

    def test_parse_opml_extracts_correct_number_of_podcasts(self, sample_opml_content: str) -> None:
        """Parse OPML extracts correct number of podcasts."""
        result = parse_opml(sample_opml_content)

        # Sample has 2 podcasts
        assert len(result["podcasts"]) == 2

    def test_parse_opml_extracts_correct_number_of_playlists(
        self, sample_opml_content: str
    ) -> None:
        """Parse OPML extracts correct number of playlists."""
        result = parse_opml(sample_opml_content)

        # Sample has 2 playlists
        assert len(result["playlists"]) == 2

    def test_parse_opml_podcast_contains_required_fields(self, sample_opml_content: str) -> None:
        """Parse OPML podcast dict contains required fields."""
        result = parse_opml(sample_opml_content)
        podcast = result["podcasts"][0]

        # Check AC2.2 - podcast has required fields
        assert "overcast_id" in podcast
        assert "title" in podcast
        assert "xml_url" in podcast
        assert "html_url" in podcast
        assert "episodes" in podcast

    def test_parse_opml_podcast_overcast_id_correct(self, sample_opml_content: str) -> None:
        """Parse OPML extracts correct overcast IDs for podcasts."""
        result = parse_opml(sample_opml_content)

        podcast_ids = {p["overcast_id"] for p in result["podcasts"]}
        assert "2001" in podcast_ids
        assert "2002" in podcast_ids

    def test_parse_opml_episode_contains_required_fields(self, sample_opml_content: str) -> None:
        """Parse OPML episode dict contains required fields."""
        result = parse_opml(sample_opml_content)
        podcast = result["podcasts"][0]
        episode = podcast["episodes"][0]

        # Check AC2.3 - episode has required fields
        assert "overcast_id" in episode
        assert "title" in episode
        assert "pub_date" in episode
        assert "enclosure_url" in episode
        assert "played" in episode or "progress" in episode

    def test_parse_opml_correct_episode_count_per_podcast(self, sample_opml_content: str) -> None:
        """Parse OPML extracts correct episode counts per podcast."""
        result = parse_opml(sample_opml_content)

        # Podcast 2001 has 4 episodes in fixture
        podcast_1 = result["podcasts"][0]
        assert len(podcast_1["episodes"]) == 4

        # Podcast 2002 has 2 episodes in fixture
        podcast_2 = result["podcasts"][1]
        assert len(podcast_2["episodes"]) == 2

    def test_parse_opml_extracts_played_attribute(self, sample_opml_content: str) -> None:
        """Parse OPML extracts played attribute when present."""
        result = parse_opml(sample_opml_content)
        podcast = result["podcasts"][0]

        # Episode 1 has played="1"
        episode_1 = podcast["episodes"][0]
        assert episode_1["played"] == "1"

        # Episode 3 has no played attribute
        episode_3 = podcast["episodes"][2]
        assert episode_3["played"] is None

    def test_parse_opml_extracts_progress_attribute(self, sample_opml_content: str) -> None:
        """Parse OPML extracts progress attribute when present."""
        result = parse_opml(sample_opml_content)
        podcast = result["podcasts"][0]

        # Episode 2 has progress="120"
        episode_2 = podcast["episodes"][1]
        assert episode_2["progress"] == "120"

        # Episode 1 has no progress attribute
        episode_1 = podcast["episodes"][0]
        assert episode_1["progress"] is None

    def test_parse_opml_extracts_playlist_episode_ids(self, sample_opml_content: str) -> None:
        """Parse OPML extracts episode IDs from playlists."""
        result = parse_opml(sample_opml_content)
        playlists = result["playlists"]

        # Check first playlist (sortedEpisodeIds)
        assert playlists[0]["episode_ids"] == ["1001", "1002", "1003"]

        # Check second playlist (includeEpisodeIds)
        assert playlists[1]["episode_ids"] == ["1001", "1004"]


class TestOpmlPodcastToMassPodcast:
    """Tests for opml_podcast_to_mass_podcast() function."""

    def test_podcast_conversion_sets_item_id(self, sample_opml_content: str) -> None:
        """Podcast conversion sets item_id to overcast_id."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        podcast = opml_podcast_to_mass_podcast(
            podcast_data, domain="overcast", instance_id="overcast_test"
        )

        # Check AC2.2 - item_id set to overcast_id
        assert podcast.item_id == "2001"

    def test_podcast_conversion_sets_name(self, sample_opml_content: str) -> None:
        """Podcast conversion sets name from title."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        podcast = opml_podcast_to_mass_podcast(
            podcast_data, domain="overcast", instance_id="overcast_test"
        )

        # Check AC2.2 - name matches title
        assert podcast.name == "Podcast One"

    def test_podcast_conversion_sets_provider_mapping(self, sample_opml_content: str) -> None:
        """Podcast conversion creates provider mapping with xml_url."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        podcast = opml_podcast_to_mass_podcast(
            podcast_data, domain="overcast", instance_id="overcast_test"
        )

        # Check AC2.2 - provider mapping includes xml_url
        assert len(podcast.provider_mappings) == 1
        mapping = next(iter(podcast.provider_mappings))
        assert mapping.item_id == "2001"
        assert mapping.url == "https://example.com/podcast1/feed/"

    def test_podcast_conversion_sets_total_episodes(self, sample_opml_content: str) -> None:
        """Podcast conversion sets total_episodes count."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        podcast = opml_podcast_to_mass_podcast(
            podcast_data, domain="overcast", instance_id="overcast_test"
        )

        # Podcast One has 4 episodes
        assert podcast.total_episodes == 4


class TestOpmlEpisodeToMassEpisode:
    """Tests for opml_episode_to_mass_episode() function."""

    def test_episode_conversion_returns_none_without_enclosure_url(
        self, sample_opml_content: str
    ) -> None:
        """Episode conversion returns None if enclosure_url missing."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]
        episode_data = podcast_data["episodes"][3]  # Episode without enclosure

        result = opml_episode_to_mass_episode(
            episode_data,
            podcast_data,
            episode_position=4,
            domain="overcast",
            instance_id="overcast_test",
        )

        assert result is None

    def test_episode_conversion_sets_item_id(self, sample_opml_content: str) -> None:
        """Episode conversion sets item_id to overcast_id."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]
        episode_data = podcast_data["episodes"][0]

        episode = opml_episode_to_mass_episode(
            episode_data,
            podcast_data,
            episode_position=1,
            domain="overcast",
            instance_id="overcast_test",
        )

        assert episode is not None
        # Check AC2.3 - item_id set to overcast_id
        assert episode.item_id == "1001"

    def test_episode_conversion_sets_name(self, sample_opml_content: str) -> None:
        """Episode conversion sets name from title."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]
        episode_data = podcast_data["episodes"][0]

        episode = opml_episode_to_mass_episode(
            episode_data,
            podcast_data,
            episode_position=1,
            domain="overcast",
            instance_id="overcast_test",
        )

        assert episode is not None
        assert episode.name == "Episode 1 - Fully Played"

    def test_episode_conversion_sets_provider_mapping(self, sample_opml_content: str) -> None:
        """Episode conversion creates provider mapping with enclosure URL."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]
        episode_data = podcast_data["episodes"][0]

        episode = opml_episode_to_mass_episode(
            episode_data,
            podcast_data,
            episode_position=1,
            domain="overcast",
            instance_id="overcast_test",
        )

        assert episode is not None
        # Check AC2.3 - provider mapping includes enclosure_url
        assert len(episode.provider_mappings) == 1
        mapping = next(iter(episode.provider_mappings))
        assert mapping.url == "https://example.com/ep1.mp3"
        assert mapping.audio_format is not None

    def test_episode_conversion_sets_podcast_reference(self, sample_opml_content: str) -> None:
        """Episode conversion sets podcast reference."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]
        episode_data = podcast_data["episodes"][0]

        episode = opml_episode_to_mass_episode(
            episode_data,
            podcast_data,
            episode_position=1,
            domain="overcast",
            instance_id="overcast_test",
        )

        assert episode is not None
        assert episode.podcast.item_id == "2001"
        assert episode.podcast.name == "Podcast One"

    def test_episode_conversion_sets_fully_played_true_when_played_is_1(
        self, sample_opml_content: str
    ) -> None:
        """Episode conversion sets fully_played=True when played='1'."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]
        episode_data = podcast_data["episodes"][0]  # Episode with played="1"

        episode = opml_episode_to_mass_episode(
            episode_data,
            podcast_data,
            episode_position=1,
            domain="overcast",
            instance_id="overcast_test",
        )

        assert episode is not None
        # Check AC2.4 - fully_played set to True when played="1"
        assert episode.fully_played is True

    def test_episode_conversion_does_not_set_fully_played_when_no_played(
        self, sample_opml_content: str
    ) -> None:
        """Episode conversion doesn't set fully_played when played attribute absent."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]
        episode_data = podcast_data["episodes"][2]  # Episode without played

        episode = opml_episode_to_mass_episode(
            episode_data,
            podcast_data,
            episode_position=3,
            domain="overcast",
            instance_id="overcast_test",
        )

        assert episode is not None
        # Check AC2.4 - fully_played not set (None) when played absent
        assert episode.fully_played is None

    def test_episode_conversion_sets_resume_position_from_progress(
        self, sample_opml_content: str
    ) -> None:
        """Episode conversion sets resume_position_ms from progress in seconds."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]
        episode_data = podcast_data["episodes"][1]  # Episode with progress="120"

        episode = opml_episode_to_mass_episode(
            episode_data,
            podcast_data,
            episode_position=2,
            domain="overcast",
            instance_id="overcast_test",
        )

        assert episode is not None
        # Check AC2.4 - resume_position_ms = progress * 1000
        assert episode.resume_position_ms == 120000

    def test_episode_conversion_parses_pub_date(self, sample_opml_content: str) -> None:
        """Episode conversion parses pub_date as ISO 8601 datetime."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]
        episode_data = podcast_data["episodes"][0]

        episode = opml_episode_to_mass_episode(
            episode_data,
            podcast_data,
            episode_position=1,
            domain="overcast",
            instance_id="overcast_test",
        )

        assert episode is not None
        assert episode.metadata.release_date is not None
        assert episode.metadata.release_date.year == 2024
        assert episode.metadata.release_date.month == 3
        assert episode.metadata.release_date.day == 20

    def test_episode_conversion_content_type_from_enclosure_url(
        self, sample_opml_content: str
    ) -> None:
        """Episode conversion extracts content type from enclosure URL."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]
        episode_data = podcast_data["episodes"][0]

        episode = opml_episode_to_mass_episode(
            episode_data,
            podcast_data,
            episode_position=1,
            domain="overcast",
            instance_id="overcast_test",
        )

        assert episode is not None
        mapping = next(iter(episode.provider_mappings))
        assert mapping.audio_format is not None
        # MP3 file should parse as MP3
        assert mapping.audio_format.content_type == ContentType.MP3


class TestEnrichEpisodesFromRss:
    """Tests for enrich_episodes_from_rss() function."""

    @staticmethod
    def _create_mock_rss_data(
        episode_count: int = 2,
        podcast_cover: str | None = None,
        episode_cover: str | None = None,
        include_descriptions: bool = True,
        include_durations: bool = True,
    ) -> dict[str, Any]:
        """
        Create mock RSS feed data matching podcastparser output format.

        :param episode_count: Number of episodes to create.
        :param podcast_cover: Podcast cover URL.
        :param episode_cover: Episode cover URL (episode_art_url).
        :param include_descriptions: Whether to include episode descriptions.
        :param include_durations: Whether to include episode durations.
        :return: Mock RSS parsed dict.
        """
        episodes: list[dict[str, Any]] = []
        for i in range(1, episode_count + 1):
            episode: dict[str, Any] = {
                "title": f"RSS Episode {i}",
                "enclosures": [{"url": f"https://example.com/ep{i}.mp3"}],
            }
            if include_descriptions:
                episode["description"] = f"Description for episode {i}"
            if include_durations:
                episode["total_time"] = 3600 + (i * 60)  # 1h + i minutes
            if episode_cover:
                episode["episode_art_url"] = episode_cover
            episodes.append(episode)

        rss_data: dict[str, Any] = {"episodes": episodes}
        if podcast_cover:
            rss_data["cover_url"] = podcast_cover
        return rss_data

    def test_enrich_episodes_from_rss_adds_description(self, sample_opml_content: str) -> None:
        """Enrich adds description from matching RSS episode."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        rss_data = self._create_mock_rss_data(
            episode_count=2,
            include_descriptions=True,
            include_durations=False,
        )
        # Match the enclosure URLs from sample OPML
        rss_data["episodes"][0]["enclosures"] = [{"url": "https://example.com/ep1.mp3"}]
        rss_data["episodes"][1]["enclosures"] = [{"url": "https://example.com/ep2.mp3"}]

        enrich_episodes_from_rss(podcast_data, rss_data)

        # Check first episode has description
        assert podcast_data["episodes"][0].get("description") == "Description for episode 1"
        # Check second episode has description
        assert podcast_data["episodes"][1].get("description") == "Description for episode 2"

    def test_enrich_episodes_from_rss_adds_duration(self, sample_opml_content: str) -> None:
        """Enrich adds duration from matching RSS episode."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        rss_data = self._create_mock_rss_data(
            episode_count=2,
            include_descriptions=False,
            include_durations=True,
        )
        rss_data["episodes"][0]["enclosures"] = [{"url": "https://example.com/ep1.mp3"}]
        rss_data["episodes"][1]["enclosures"] = [{"url": "https://example.com/ep2.mp3"}]

        enrich_episodes_from_rss(podcast_data, rss_data)

        # Check first episode has duration
        assert podcast_data["episodes"][0].get("duration") == 3660  # 3600 + 60
        # Check second episode has duration
        assert podcast_data["episodes"][1].get("duration") == 3720  # 3600 + 120

    def test_enrich_episodes_from_rss_uses_episode_cover_url(
        self, sample_opml_content: str
    ) -> None:
        """Enrich prefers episode cover art over podcast cover."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        rss_data = self._create_mock_rss_data(
            episode_count=1,
            podcast_cover="https://example.com/podcast.jpg",
            episode_cover="https://example.com/ep1_cover.jpg",
        )
        rss_data["episodes"][0]["enclosures"] = [{"url": "https://example.com/ep1.mp3"}]

        enrich_episodes_from_rss(podcast_data, rss_data)

        # Check episode has episode cover art (not podcast cover)
        assert podcast_data["episodes"][0].get("cover_url") == "https://example.com/ep1_cover.jpg"

    def test_enrich_episodes_from_rss_uses_podcast_cover_as_fallback(
        self, sample_opml_content: str
    ) -> None:
        """Enrich falls back to podcast cover when episode cover unavailable."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        rss_data = self._create_mock_rss_data(
            episode_count=1,
            podcast_cover="https://example.com/podcast.jpg",
            episode_cover=None,
        )
        rss_data["episodes"][0]["enclosures"] = [{"url": "https://example.com/ep1.mp3"}]

        enrich_episodes_from_rss(podcast_data, rss_data)

        # Check episode has podcast cover as fallback
        assert podcast_data["episodes"][0].get("cover_url") == "https://example.com/podcast.jpg"

    def test_enrich_episodes_from_rss_preserves_unenriched_episodes(
        self, sample_opml_content: str
    ) -> None:
        """Enrich leaves unmatched OPML episodes unchanged (AC3.3)."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        # RSS only has one episode (first one)
        rss_data = self._create_mock_rss_data(episode_count=1)
        rss_data["episodes"][0]["enclosures"] = [{"url": "https://example.com/ep1.mp3"}]

        # Store original data for comparison
        original_ep2_title = podcast_data["episodes"][1]["title"]
        original_ep2_enclosure = podcast_data["episodes"][1]["enclosure_url"]

        enrich_episodes_from_rss(podcast_data, rss_data)

        # First episode should be enriched
        assert "description" in podcast_data["episodes"][0]
        # Second episode should have no enrichment (no description)
        assert "description" not in podcast_data["episodes"][1]
        # But original data preserved
        assert podcast_data["episodes"][1]["title"] == original_ep2_title
        assert podcast_data["episodes"][1]["enclosure_url"] == original_ep2_enclosure

    def test_enrich_episodes_from_rss_handles_empty_rss_dict(
        self, sample_opml_content: str
    ) -> None:
        """Enrich handles empty RSS dict gracefully (no episodes)."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        # Store original episode count
        original_episode_count = len(podcast_data["episodes"])

        empty_rss_data: dict[str, Any] = {"episodes": []}

        enrich_episodes_from_rss(podcast_data, empty_rss_data)

        # Podcast should still have same episodes
        assert len(podcast_data["episodes"]) == original_episode_count
        # But none should be enriched
        for episode in podcast_data["episodes"]:
            assert "description" not in episode
            assert "duration" not in episode

    def test_enrich_episodes_from_rss_with_authenticated_urls(
        self, sample_opml_content: str
    ) -> None:
        """Enrich works with authenticated feed URLs (AC3.2)."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        # Create RSS data where enclosure URL has auth params
        rss_data: dict[str, Any] = {
            "episodes": [
                {
                    "title": "Episode 1",
                    "enclosures": [
                        {"url": "https://cdn.example.com/ep1.mp3?token=abc123&expires=1234567890"}
                    ],
                    "description": "Authenticated episode",
                    "total_time": 3600,
                }
            ]
        }

        # Simulate authenticated OPML podcast URL
        podcast_data["xml_url"] = "https://feeds.example.com/feed?token=xyz789"

        # Update OPML episode to match authenticated enclosure URL
        podcast_data["episodes"][0]["enclosure_url"] = (
            "https://cdn.example.com/ep1.mp3?token=abc123&expires=1234567890"
        )

        enrich_episodes_from_rss(podcast_data, rss_data)

        # Episode should be enriched even with authenticated URL
        assert podcast_data["episodes"][0].get("description") == "Authenticated episode"
        assert podcast_data["episodes"][0].get("duration") == 3600
        # Authenticated podcast URL should be preserved in podcast_data
        assert podcast_data["xml_url"] == "https://feeds.example.com/feed?token=xyz789"

    def test_enrich_episodes_from_rss_rejects_non_image_cover_url(
        self, sample_opml_content: str
    ) -> None:
        """
        Non-image URLs (XML feeds, HTML pages) are not used as cover art.

        Some RSS feeds return their feed URL as cover_url instead of an actual
        image. These should be filtered out to prevent PIL errors downstream.
        """
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        rss_data: dict[str, Any] = {
            "cover_url": "https://example.com/episode/index.xml",
            "episodes": [
                {
                    "title": "Episode 1",
                    "enclosures": [{"url": podcast_data["episodes"][0]["enclosure_url"]}],
                    "description": "Test",
                    "total_time": 100,
                    "episode_art_url": None,
                }
            ],
        }

        enrich_episodes_from_rss(podcast_data, rss_data)

        # Podcast cover should NOT be set (it's an XML feed URL)
        assert "cover_url" not in podcast_data
        # Episode cover should NOT be set either
        assert "cover_url" not in podcast_data["episodes"][0]

    def test_enriched_episode_conversion_uses_all_fields(self, sample_opml_content: str) -> None:
        """Episode conversion uses all enriched fields (duration, description, cover)."""
        parsed = parse_opml(sample_opml_content)
        podcast_data = parsed["podcasts"][0]

        # Enrich episodes with RSS data
        rss_data = self._create_mock_rss_data(
            episode_count=1,
            podcast_cover="https://example.com/podcast.jpg",
            episode_cover="https://example.com/ep1_cover.jpg",
        )
        rss_data["episodes"][0]["enclosures"] = [{"url": "https://example.com/ep1.mp3"}]

        enrich_episodes_from_rss(podcast_data, rss_data)

        # Now convert enriched episode to Mass episode
        episode_data = podcast_data["episodes"][0]
        episode = opml_episode_to_mass_episode(
            episode_data,
            podcast_data,
            episode_position=1,
            domain="overcast",
            instance_id="overcast_test",
        )

        assert episode is not None
        # Check all enriched fields are set
        assert episode.duration == 3660  # From RSS
        assert episode.metadata.description == "Description for episode 1"  # From RSS
        assert episode.metadata.images is not None
        assert len(episode.metadata.images) > 0  # Cover image set
        cover_image = episode.metadata.images[0]
        assert cover_image.path == "https://example.com/ep1_cover.jpg"


class TestParseEpisodePage:
    """Tests for parse_episode_page() function."""

    def test_parse_episode_page_extracts_valid_html_with_both_attributes(self) -> None:
        """Parse episode page extracts both data-start-time and data-sync-version."""
        html = '<audio id="audioplayer" data-start-time="120" data-sync-version="5">'
        start_time, sync_version = parse_episode_page(html)

        assert start_time == 120
        assert sync_version == 5

    def test_parse_episode_page_returns_none_for_missing_attributes(self) -> None:
        """Parse episode page returns None for missing attributes."""
        html = '<audio id="audioplayer">'
        start_time, sync_version = parse_episode_page(html)

        assert start_time is None
        assert sync_version is None

    def test_parse_episode_page_returns_none_for_no_audio_player(self) -> None:
        """Parse episode page returns None when no audio player element."""
        html = '<div id="content"><p>No audio player here</p></div>'
        start_time, sync_version = parse_episode_page(html)

        assert start_time is None
        assert sync_version is None

    def test_parse_episode_page_handles_only_start_time(self) -> None:
        """Parse episode page handles HTML with only data-start-time."""
        html = '<audio id="audioplayer" data-start-time="300">'
        start_time, sync_version = parse_episode_page(html)

        assert start_time == 300
        assert sync_version is None

    def test_parse_episode_page_handles_only_sync_version(self) -> None:
        """Parse episode page handles HTML with only data-sync-version."""
        html = '<audio id="audioplayer" data-sync-version="42">'
        start_time, sync_version = parse_episode_page(html)

        assert start_time is None
        assert sync_version == 42

    def test_parse_episode_page_handles_attributes_in_different_order(self) -> None:
        """Parse episode page handles attributes regardless of order."""
        html = '<audio id="audioplayer" data-sync-version="7" data-start-time="250">'
        start_time, sync_version = parse_episode_page(html)

        assert start_time == 250
        assert sync_version == 7

    def test_parse_episode_page_handles_single_quotes(self) -> None:
        """Single-quoted attributes are valid HTML and must be supported."""
        html = "<audio id='audioplayer' data-start-time='100' data-sync-version='10'>"
        start_time, sync_version = parse_episode_page(html)

        assert start_time == 100
        assert sync_version == 10

    def test_parse_episode_page_ignores_attrs_outside_audioplayer(self) -> None:
        """
        Attributes outside the #audioplayer element must not be matched.

        A previous regex matched anywhere in the document — a comment or
        unrelated element mentioning data-start-time would fool it.
        """
        html = '<!-- data-start-time="9999" --><div data-sync-version="42">unrelated</div>'
        start_time, sync_version = parse_episode_page(html)

        assert start_time is None
        assert sync_version is None


class TestParsingEdgeCases:
    """Tests for parsing edge cases (missing data, malformed input)."""

    def test_opml_episode_to_mass_episode_returns_none_for_no_enclosure_url(self) -> None:
        """
        Episodes with no enclosureUrl are skipped (AC8.4).

        opml_episode_to_mass_episode() should return None for episodes
        that have no enclosure_url, so they are skipped in iteration.
        """
        episode_data = {
            "overcast_id": "3004",
            "title": "Episode 4",
            "pub_date": "2024-10-09T16:00:00-04:00",
            "enclosure_url": None,
            "overcast_url": "https://overcast.fm/+ABC126",
            "played": None,
            "progress": None,
            "user_deleted": None,
        }
        podcast_data = {
            "overcast_id": "2001",
            "title": "Podcast One",
            "html_url": "https://example.com/podcast1",
        }

        result = opml_episode_to_mass_episode(
            episode_data=episode_data,
            podcast_data=podcast_data,
            episode_position=3,
            domain="overcast",
            instance_id="overcast_test",
        )

        # Should return None because no enclosure_url
        assert result is None

    def test_parse_opml_includes_episodes_without_enclosure_in_raw_data(
        self, sample_opml_content: str
    ) -> None:
        """
        parse_opml() includes episodes without enclosure in raw data.

        Even though episodes without enclosureUrl exist in the OPML,
        parse_opml() should include them in the raw data. Filtering happens
        in opml_episode_to_mass_episode() when converting to mass episodes.
        """
        result = parse_opml(sample_opml_content)

        # Podcast 2001 has 4 episodes including one without enclosure_url
        podcast_1 = result["podcasts"][0]
        assert len(podcast_1["episodes"]) == 4

        # Episode 1004 has no enclosure_url (last episode in list)
        episode_4 = podcast_1["episodes"][3]
        assert episode_4.get("overcast_id") == "1004"
        assert episode_4.get("enclosure_url") is None
