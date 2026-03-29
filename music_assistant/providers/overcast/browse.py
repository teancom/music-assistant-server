"""Browse tree implementation for Overcast provider."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from music_assistant_models.media_items import BrowseFolder, PodcastEpisode

from .constants import BROWSE_PLAYLISTS, BROWSE_PODCASTS
from .parsing import opml_episode_to_mass_episode


def build_episode_lookup(
    opml_data: dict[str, Any],
) -> dict[str, tuple[dict[str, Any], dict[str, Any], int]]:
    """
    Build a mapping from episode overcastId to (episode_data, podcast_data, position).

    Creates a lookup table for efficient episode resolution across all podcasts,
    used when browsing playlists.

    :param opml_data: The parsed OPML data dict.
    :return: Dict mapping episode overcastId to (episode_data, podcast_data, position).
    """
    lookup: dict[str, tuple[dict[str, Any], dict[str, Any], int]] = {}
    for podcast_data in opml_data.get("podcasts", []):
        for position, episode_data in enumerate(podcast_data.get("episodes", [])):
            episode_id = episode_data.get("overcast_id")
            if episode_id:
                lookup[episode_id] = (episode_data, podcast_data, position)
    return lookup


def browse_root(base: str, domain: str) -> list[BrowseFolder]:
    """
    Return root browse folders (Playlists and Podcasts).

    :param base: The base path for the instance.
    :param domain: Provider domain.
    :return: List of two BrowseFolder items.
    """
    return [
        BrowseFolder(
            item_id=BROWSE_PLAYLISTS,
            provider=domain,
            path=f"{base}{BROWSE_PLAYLISTS}",
            name="Playlists",
            translation_key="playlists",
        ),
        BrowseFolder(
            item_id=BROWSE_PODCASTS,
            provider=domain,
            path=f"{base}{BROWSE_PODCASTS}",
            name="Podcasts",
            translation_key="podcasts",
        ),
    ]


def browse_playlists_list(base: str, domain: str, opml_data: dict[str, Any]) -> list[BrowseFolder]:
    """
    Return list of all playlists as BrowseFolder items.

    :param base: The base path for the instance.
    :param domain: Provider domain.
    :param opml_data: The parsed OPML data dict.
    :return: List of BrowseFolder items for each playlist.
    """
    result: list[BrowseFolder] = []
    for playlist in opml_data.get("playlists", []):
        playlist_title = playlist.get("title", "Unknown")
        is_smart = playlist.get("smart") == "1"
        name = f"{playlist_title} (Smart)" if is_smart else playlist_title
        result.append(
            BrowseFolder(
                item_id=playlist_title,
                provider=domain,
                path=f"{base}{BROWSE_PLAYLISTS}/{quote(playlist_title, safe='')}",
                name=name,
            )
        )
    return result


def browse_playlist_episodes(
    playlist_title: str,
    opml_data: dict[str, Any],
    episode_lookup: dict[str, tuple[dict[str, Any], dict[str, Any], int]],
    domain: str,
    instance_id: str,
) -> list[PodcastEpisode]:
    """
    Return episodes for a specific playlist.

    :param playlist_title: Title of the playlist to browse.
    :param opml_data: The parsed OPML data dict.
    :param episode_lookup: Mapping of episode ID to (episode, podcast, position) tuples.
    :param domain: Provider domain.
    :param instance_id: Provider instance ID.
    :return: List of PodcastEpisode items.
    """
    playlist = next(
        (p for p in opml_data.get("playlists", []) if p.get("title") == playlist_title),
        None,
    )
    if playlist is None:
        return []

    episodes_result: list[PodcastEpisode] = []
    for episode_id in playlist.get("episode_ids", []):
        if episode_id not in episode_lookup:
            continue
        episode_data, podcast_data, episode_position = episode_lookup[episode_id]
        mass_episode = opml_episode_to_mass_episode(
            episode_data=episode_data,
            podcast_data=podcast_data,
            episode_position=episode_position,
            domain=domain,
            instance_id=instance_id,
        )
        if mass_episode is not None:
            episodes_result.append(mass_episode)
    return episodes_result


def browse_podcasts_list(base: str, domain: str, opml_data: dict[str, Any]) -> list[BrowseFolder]:
    """
    Return list of all podcasts as BrowseFolder items.

    :param base: The base path for the instance.
    :param domain: Provider domain.
    :param opml_data: The parsed OPML data dict.
    :return: List of BrowseFolder items for each podcast.
    """
    result: list[BrowseFolder] = []
    for podcast_data in opml_data.get("podcasts", []):
        podcast_id = podcast_data.get("overcast_id")
        podcast_title = podcast_data.get("title", "Unknown")
        result.append(
            BrowseFolder(
                item_id=podcast_id,
                provider=domain,
                path=f"{base}{BROWSE_PODCASTS}/{podcast_id}",
                name=podcast_title,
            )
        )
    return result


def browse_podcast_episodes(
    podcast_id: str,
    opml_data: dict[str, Any],
    domain: str,
    instance_id: str,
    unplayed_only: bool = False,
) -> list[PodcastEpisode]:
    """
    Return episodes for a specific podcast, optionally filtered to unplayed.

    :param podcast_id: The overcast_id of the podcast to browse.
    :param opml_data: The parsed OPML data dict.
    :param domain: Provider domain.
    :param instance_id: Provider instance ID.
    :param unplayed_only: If True, filter to unplayed episodes only.
    :return: List of PodcastEpisode items.
    """
    podcast_data = next(
        (p for p in opml_data.get("podcasts", []) if p.get("overcast_id") == podcast_id),
        None,
    )
    if podcast_data is None:
        return []

    episodes_result: list[PodcastEpisode] = []
    for episode_position, episode_data in enumerate(podcast_data.get("episodes", [])):
        if unplayed_only and episode_data.get("played") == "1":
            continue
        mass_episode = opml_episode_to_mass_episode(
            episode_data=episode_data,
            podcast_data=podcast_data,
            episode_position=episode_position,
            domain=domain,
            instance_id=instance_id,
        )
        if mass_episode is not None:
            episodes_result.append(mass_episode)

    return episodes_result
