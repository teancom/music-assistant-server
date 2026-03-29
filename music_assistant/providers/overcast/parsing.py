"""
Parsing helpers for Overcast provider.

Custom OPML parsing instead of a library (e.g. listparser) because Overcast's OPML
uses non-standard attributes (overcastId, overcastUrl, played, progress, etc.)
that generic OPML parsers discard. Also handles RSS enrichment, HTML page parsing,
and conversion of parsed data to Music Assistant model objects.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from defusedxml.ElementTree import fromstring as safe_fromstring
from music_assistant_models.enums import ContentType, ImageType, MediaType
from music_assistant_models.media_items import (
    AudioFormat,
    ItemMapping,
    MediaItemImage,
    Podcast,
    PodcastEpisode,
    ProviderMapping,
)
from music_assistant_models.unique_list import UniqueList

from .constants import IMAGE_EXTENSIONS

if TYPE_CHECKING:
    from xml.etree import ElementTree as ET

# Match the audio player element first, then look for the data attributes
# *inside* its tag. Scoping prevents accidental hits on comments or other
# elements that might mention the attributes verbatim.
_AUDIOPLAYER_RE = re.compile(
    r"""<[^>]*\bid=['"]audioplayer['"][^>]*>""",
    re.IGNORECASE,
)
_START_TIME_RE = re.compile(r"""\bdata-start-time=['"](\d+)['"]""")
_SYNC_VERSION_RE = re.compile(r"""\bdata-sync-version=['"](\d+)['"]""")


def parse_opml(opml_text: str) -> dict[str, Any]:
    """
    Parse OPML XML text and return structured dict with podcasts and playlists.

    :param opml_text: OPML XML content as string.
    :return: Dict with keys "podcasts" (list) and "playlists" (list).
    """
    root = safe_fromstring(opml_text)
    body = root.find("body")
    if body is None:
        return {"podcasts": [], "playlists": []}

    playlists_section = body.find("outline[@text='playlists']")
    playlist_outlines = (
        playlists_section.findall("outline[@type='podcast-playlist']")
        if playlists_section is not None
        else []
    )
    playlists = [_parse_playlist_outline(el) for el in playlist_outlines]

    feeds_section = body.find("outline[@text='feeds']")
    feed_outlines = (
        feeds_section.findall("outline[@type='rss']") if feeds_section is not None else []
    )
    podcasts = [_parse_podcast_outline(el) for el in feed_outlines]

    return {"podcasts": podcasts, "playlists": playlists}


def _parse_playlist_outline(outline: ET.Element) -> dict[str, Any]:
    """
    Parse a single playlist outline element into a dict.

    :param outline: XML element for a podcast-playlist.
    :return: Dict with title, smart flag, and episode IDs.
    """
    return {
        "title": outline.get("title", ""),
        "smart": outline.get("smart", "0"),
        "episode_ids": _parse_episode_ids(outline),
    }


def _parse_podcast_outline(outline: ET.Element) -> dict[str, Any]:
    """
    Parse a single podcast outline element and its episodes into a dict.

    :param outline: XML element for an RSS podcast feed.
    :return: Dict with podcast metadata and episodes list.
    """
    return {
        "overcast_id": outline.get("overcastId", ""),
        "title": outline.get("title", ""),
        "xml_url": outline.get("xmlUrl", ""),
        "html_url": outline.get("htmlUrl", ""),
        "episodes": [
            _parse_episode_outline(ep) for ep in outline.findall("outline[@type='podcast-episode']")
        ],
    }


def _parse_episode_outline(outline: ET.Element) -> dict[str, Any]:
    """
    Parse a single episode outline element into a dict.

    :param outline: XML element for a podcast-episode.
    :return: Dict with episode metadata.
    """
    return {
        "overcast_id": outline.get("overcastId", ""),
        "title": outline.get("title", ""),
        "pub_date": outline.get("pubDate", ""),
        "enclosure_url": outline.get("enclosureUrl"),
        "overcast_url": outline.get("overcastUrl", ""),
        "played": outline.get("played"),
        "progress": outline.get("progress"),
        "user_deleted": outline.get("userDeleted"),
    }


def parse_episode_page(html: str) -> tuple[int | None, int | None]:
    """
    Extract resume state from an Overcast episode web page.

    Reads ``data-start-time`` and ``data-sync-version`` from the
    ``id="audioplayer"`` opening tag. Either value may be missing; both
    return ``None`` if the audio player element itself is absent.

    :param html: Raw HTML of the Overcast episode page.
    :return: ``(start_time_seconds, sync_version)``.
    """
    audio_match = _AUDIOPLAYER_RE.search(html)
    if audio_match is None:
        return None, None
    audio_tag = audio_match.group(0)
    start_match = _START_TIME_RE.search(audio_tag)
    start_time = int(start_match.group(1)) if start_match else None
    version_match = _SYNC_VERSION_RE.search(audio_tag)
    sync_version = int(version_match.group(1)) if version_match else None
    return start_time, sync_version


def enrich_episodes_from_rss(podcast_data: dict[str, Any], rss_parsed: dict[str, Any]) -> None:
    """
    Enrich OPML episodes with data from RSS feed.

    Matches OPML episodes to RSS episodes by enclosure URL and merges enrichment
    fields (description, duration, cover_url). Also stores the podcast-level cover
    URL on podcast_data. Operates in-place on podcast_data.

    :param podcast_data: Podcast dict from parse_opml() with episodes list.
    :param rss_parsed: Parsed RSS feed dict from podcastparser.parse().
    """
    podcast_cover = rss_parsed.get("cover_url")
    if podcast_cover and _is_image_url(podcast_cover):
        podcast_data["cover_url"] = podcast_cover

    # Build lookup dict from RSS episodes by enclosure URL
    rss_episodes_by_url: dict[str, dict[str, Any]] = {}
    for rss_episode in rss_parsed.get("episodes", []):
        enclosures = rss_episode.get("enclosures", [])
        if enclosures:
            enclosure_url = enclosures[0].get("url")
            if enclosure_url:
                rss_episodes_by_url[enclosure_url] = rss_episode

    # Enrich OPML episodes with RSS data
    for opml_episode in podcast_data.get("episodes", []):
        enclosure_url = opml_episode.get("enclosure_url")
        if not enclosure_url:
            continue

        # Look up matching RSS episode
        rss_episode = rss_episodes_by_url.get(enclosure_url)
        if not rss_episode:
            continue

        # Merge enrichment fields
        opml_episode["description"] = rss_episode.get("description", "")
        opml_episode["duration"] = rss_episode.get("total_time", 0)
        # Prefer episode cover art, fall back to podcast cover
        episode_art = rss_episode.get("episode_art_url") or rss_parsed.get("cover_url")
        if episode_art and _is_image_url(episode_art):
            opml_episode["cover_url"] = episode_art


def _parse_episode_ids(playlist_outline: ET.Element) -> list[str]:
    """
    Extract episode IDs from a playlist outline element.

    :param playlist_outline: XML element for a podcast-playlist.
    :return: List of episode IDs.
    """
    episode_ids_str = playlist_outline.get("sortedEpisodeIds") or playlist_outline.get(
        "includeEpisodeIds", ""
    )
    if not episode_ids_str:
        return []
    return [eid.strip() for eid in episode_ids_str.split(",") if eid.strip()]


def _is_image_url(url: str) -> bool:
    """
    Best-effort allowlist filter for cover-art URLs.

    Returns ``True`` for known raster image extensions and for extensionless
    URLs (CDN paths that hide the file type). Returns ``False`` for feeds,
    HTML pages, and SVG (which can carry script and is a poor fit for cover
    art rendering). Downstream image fetcher should still validate
    ``Content-Type`` — this is a cheap pre-filter, not a security boundary.

    :param url: URL to check.
    """
    path = url.lower().split("?")[0].split("#")[0]
    last_segment = path.rsplit("/", 1)[-1]
    if "." not in last_segment:
        # Extensionless URLs (e.g. CDN paths) — accept; Content-Type validates downstream.
        return True
    return path.endswith(IMAGE_EXTENSIONS)


def _make_thumbnail(cover_url: str, instance_id: str) -> UniqueList[MediaItemImage]:
    """
    Create a thumbnail image list from a cover URL.

    :param cover_url: URL of the cover image.
    :param instance_id: Provider instance ID.
    :return: UniqueList with a single THUMB MediaItemImage.
    """
    return UniqueList(
        [
            MediaItemImage(
                type=ImageType.THUMB,
                path=cover_url,
                provider=instance_id,
                remotely_accessible=True,
            )
        ]
    )


def opml_podcast_to_mass_podcast(
    podcast_data: dict[str, Any], domain: str, instance_id: str
) -> Podcast:
    """
    Convert a parsed OPML podcast dict to a Music Assistant ``Podcast``.

    :param podcast_data: Podcast dict from ``parse_opml()``.
    :param domain: Provider domain (used for the ``ProviderMapping``).
    :param instance_id: Provider instance ID.
    :raises ValueError: If ``podcast_data`` is missing the ``overcast_id`` key.
    """
    overcast_id = podcast_data.get("overcast_id")
    if not overcast_id:
        raise ValueError("OPML podcast entry is missing overcast_id")
    title = podcast_data.get("title") or ""
    xml_url = podcast_data.get("xml_url", "")

    mass_podcast = Podcast(
        item_id=overcast_id,
        name=title,
        provider=instance_id,
        uri=podcast_data.get("html_url"),
        provider_mappings={
            ProviderMapping(
                item_id=overcast_id,
                provider_domain=domain,
                provider_instance=instance_id,
                url=xml_url,
            )
        },
    )

    # Set total episodes count
    episodes = podcast_data.get("episodes", [])
    mass_podcast.total_episodes = len(episodes)

    cover_url = podcast_data.get("cover_url")
    if cover_url:
        mass_podcast.metadata.images = _make_thumbnail(cover_url, instance_id)

    return mass_podcast


def opml_episode_to_mass_episode(
    episode_data: dict[str, Any],
    podcast_data: dict[str, Any],
    episode_position: int,
    domain: str,
    instance_id: str,
) -> PodcastEpisode | None:
    """
    Convert a parsed OPML episode dict to a ``PodcastEpisode``.

    :param episode_data: Episode dict from ``parse_opml()``.
    :param podcast_data: The parent podcast dict (used for podcast back-reference).
    :param episode_position: Zero-based index of the episode within the podcast.
    :param domain: Provider domain.
    :param instance_id: Provider instance ID.
    :return: A ``PodcastEpisode``, or ``None`` if the OPML entry has no
        enclosure URL or is missing required identifiers (cannot stream).
    """
    enclosure_url = episode_data.get("enclosure_url")
    if not enclosure_url:
        return None

    overcast_id = episode_data.get("overcast_id")
    podcast_overcast_id = podcast_data.get("overcast_id")
    if not overcast_id or not podcast_overcast_id:
        # Defensive: OPML missing overcastId means we cannot route playback or
        # progress write-back back to this episode — drop it rather than emit
        # a partial entry.
        return None
    title = episode_data.get("title") or ""

    mass_episode = PodcastEpisode(
        item_id=overcast_id,
        provider=instance_id,
        name=title,
        position=episode_position,
        podcast=ItemMapping(
            item_id=podcast_overcast_id,
            provider=instance_id,
            name=podcast_data.get("title") or "",
            media_type=MediaType.PODCAST,
        ),
        provider_mappings={
            ProviderMapping(
                item_id=overcast_id,
                provider_domain=domain,
                provider_instance=instance_id,
                url=enclosure_url,
                audio_format=AudioFormat(
                    content_type=ContentType.try_parse(enclosure_url),
                ),
            )
        },
    )

    # Set duration from enriched data
    duration = episode_data.get("duration", 0)
    if duration:
        mass_episode.duration = int(duration)

    # Set description from enriched data
    description = episode_data.get("description", "")
    if description:
        mass_episode.metadata.description = description

    cover_url = episode_data.get("cover_url")
    if cover_url:
        mass_episode.metadata.images = _make_thumbnail(cover_url, instance_id)

    # Parse and set release date
    pub_date_str = episode_data.get("pub_date")
    if pub_date_str:
        try:
            # Parse ISO 8601 datetime string
            pub_date = datetime.fromisoformat(pub_date_str)
            mass_episode.metadata.release_date = pub_date
        except ValueError, TypeError:
            pass

    # Set fully_played from "played" attribute
    if episode_data.get("played") == "1":
        mass_episode.fully_played = True

    # Set resume position from "progress" attribute (in seconds, convert to ms)
    progress_str = episode_data.get("progress")
    if progress_str:
        try:
            progress_seconds = int(progress_str)
            mass_episode.resume_position_ms = progress_seconds * 1000
        except ValueError, TypeError:
            pass

    return mass_episode
