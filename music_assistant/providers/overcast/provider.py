"""Overcast provider implementation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote

import aiohttp
from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.errors import (
    LoginFailed,
    MediaNotFoundError,
    MusicAssistantError,
    RateLimited,
    ResourceTemporarilyUnavailable,
    RetriesExhausted,
)
from music_assistant_models.media_items import (
    AudioFormat,
    BrowseFolder,
    MediaItemType,
    PodcastEpisode,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.helpers.podcast_parsers import get_podcastparser_dict
from music_assistant.helpers.throttle_retry import (
    ThrottlerManager,
    parse_retry_after,
    throttle_with_retries,
)
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant_models.media_items import Podcast

from . import browse
from .constants import (
    BROWSE_PLAYLISTS,
    BROWSE_PODCASTS,
    BROWSE_UNPLAYED,
    CACHE_CATEGORY_OPML,
    CACHE_KEY_OPML,
    CACHE_KEY_RSS_ENRICHMENT,
    CACHE_TTL,
    CACHE_TTL_ENRICHMENT,
    CONF_COOKIE,
    ERR_COOKIE_EXPIRED,
    FINISHED_SENTINEL,
    GUARD_INTERVAL_ACTION,
    GUARD_INTERVAL_PLAYING,
    OVERCAST_ACCOUNT_URL,
    OVERCAST_OPML_URL,
    OVERCAST_SET_PROGRESS_URL,
)
from .parsing import (
    enrich_episodes_from_rss,
    opml_episode_to_mass_episode,
    opml_podcast_to_mass_podcast,
    parse_episode_page,
    parse_opml,
)


class OvercastProvider(MusicProvider):
    """Overcast podcast provider for Music Assistant."""

    _cookie: str = ""
    # Per-episode write-back state. Initialized at class scope so callers
    # (including code paths that bypass handle_async_init in tests) always
    # see a well-formed dict instead of needing hasattr guards.
    _sync_versions: dict[str, int] = {}  # noqa: RUF012
    _last_progress_time: dict[str, float] = {}  # noqa: RUF012
    _opml_lock: asyncio.Lock | None = None
    throttler = ThrottlerManager(rate_limit=1, period=5, retry_attempts=3, initial_backoff=30)

    async def handle_async_init(self) -> None:
        """Validate the configured cookie and prepare per-instance state."""
        cookie = self.config.get_value(CONF_COOKIE)
        if not cookie:
            raise LoginFailed(
                "Overcast session cookie is required. "
                "Log in at https://overcast.fm, open browser DevTools, "
                "and copy the value of the cookie named 'o'."
            )
        self._cookie = str(cookie)
        # Fresh per-instance dicts — overrides the class-level defaults so
        # multiple provider instances don't share state.
        self._sync_versions = {}
        self._last_progress_time = {}
        # Per-instance lock serializing OPML fetches so concurrent callers
        # (e.g. library sync + a browse request) share a single network fetch
        # rather than each hitting Overcast and racing on the cache.
        self._opml_lock = asyncio.Lock()

        await self._validate_auth()
        self.logger.debug("Overcast session cookie validated successfully")

    async def get_library_podcasts(self) -> AsyncGenerator[Podcast]:
        """Yield ``Podcast`` items for every subscription."""
        opml_data = await self._get_opml_data()
        for podcast_data in opml_data.get("podcasts", []):
            podcast = opml_podcast_to_mass_podcast(
                podcast_data=podcast_data,
                domain=self.domain,
                instance_id=self.instance_id,
            )
            yield podcast

    async def get_podcast(self, prov_podcast_id: str) -> Podcast:
        """
        Return a ``Podcast`` by its overcastId.

        :param prov_podcast_id: The overcastId of the podcast.
        :raises MediaNotFoundError: If no podcast with that ID is subscribed.
        """
        opml_data = await self._get_opml_data()
        for podcast_data in opml_data.get("podcasts", []):
            if podcast_data.get("overcast_id") == prov_podcast_id:
                return opml_podcast_to_mass_podcast(
                    podcast_data=podcast_data,
                    domain=self.domain,
                    instance_id=self.instance_id,
                )
        raise MediaNotFoundError(f"Podcast with ID {prov_podcast_id} not found")

    async def get_podcast_episodes(self, prov_podcast_id: str) -> AsyncGenerator[PodcastEpisode]:
        """
        Yield ``PodcastEpisode`` items for a given podcast.

        Episodes without an enclosure URL are skipped (cannot stream).

        :param prov_podcast_id: The overcastId of the podcast.
        :raises MediaNotFoundError: If no podcast with that ID is subscribed.
        """
        opml_data = await self._get_opml_data()
        podcast_data = next(
            (p for p in opml_data.get("podcasts", []) if p.get("overcast_id") == prov_podcast_id),
            None,
        )
        if podcast_data is None:
            raise MediaNotFoundError(f"Podcast with ID {prov_podcast_id} not found")

        for episode_position, episode_data in enumerate(podcast_data.get("episodes", [])):
            mass_episode = opml_episode_to_mass_episode(
                episode_data=episode_data,
                podcast_data=podcast_data,
                episode_position=episode_position,
                domain=self.domain,
                instance_id=self.instance_id,
            )
            if mass_episode is not None:
                yield mass_episode

    async def get_podcast_episode(self, prov_episode_id: str) -> PodcastEpisode:
        """
        Return a ``PodcastEpisode`` by its overcastId.

        :param prov_episode_id: The overcastId of the episode.
        :raises MediaNotFoundError: If no episode with that ID exists or it
            lacks an enclosure URL.
        """
        opml_data = await self._get_opml_data()
        for podcast_data in opml_data.get("podcasts", []):
            for episode_position, episode_data in enumerate(podcast_data.get("episodes", [])):
                if episode_data.get("overcast_id") == prov_episode_id:
                    mass_episode = opml_episode_to_mass_episode(
                        episode_data=episode_data,
                        podcast_data=podcast_data,
                        episode_position=episode_position,
                        domain=self.domain,
                        instance_id=self.instance_id,
                    )
                    if mass_episode is not None:
                        return mass_episode
        raise MediaNotFoundError(f"Episode with ID {prov_episode_id} not found")

    async def get_stream_details(
        self,
        item_id: str,
        media_type: MediaType,
    ) -> StreamDetails:
        """
        Return ``StreamDetails`` for a podcast episode.

        :param item_id: The overcastId of the episode to stream.
        :param media_type: Media type (always ``PODCAST_EPISODE`` here).
        :raises MediaNotFoundError: If the episode is unknown or has no
            enclosure URL.
        """
        episode = await self._find_episode_in_opml(item_id)
        if episode is None or not episode.get("enclosure_url"):
            raise MediaNotFoundError(f"Episode {item_id} not found")
        enclosure_url = episode["enclosure_url"]
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=AudioFormat(
                content_type=ContentType.try_parse(enclosure_url),
            ),
            media_type=MediaType.PODCAST_EPISODE,
            stream_type=StreamType.HTTP,
            path=enclosure_url,
            can_seek=True,
            allow_seek=True,
        )

    async def on_played(
        self,
        media_type: MediaType,
        prov_item_id: str,
        fully_played: bool,
        position: int,
        media_item: MediaItemType,
        is_playing: bool = False,
    ) -> None:
        """
        Sync playback progress for a podcast episode to Overcast.

        Pause / stop / completion events POST within ``GUARD_INTERVAL_ACTION``
        (1s); active-playback heartbeats are throttled to ``GUARD_INTERVAL_PLAYING``
        (60s) to avoid hammering the API. ``LoginFailed`` propagates so the
        framework can prompt re-authentication; transient errors are logged
        and swallowed so playback is never interrupted.

        :param media_type: Item media type — non-podcast events are ignored.
        :param prov_item_id: The overcastId of the episode.
        :param fully_played: ``True`` when the episode was played to completion.
        :param position: Playback position in seconds.
        :param media_item: The full media item — must be a ``PodcastEpisode``.
        :param is_playing: ``True`` for heartbeats, ``False`` for pause/stop.
        """
        if media_type != MediaType.PODCAST_EPISODE:
            return
        if not isinstance(media_item, PodcastEpisode):
            return

        try:
            if fully_played:
                guard_interval = GUARD_INTERVAL_ACTION
                effective_position = FINISHED_SENTINEL
            elif not is_playing:
                guard_interval = GUARD_INTERVAL_ACTION
                effective_position = position
            else:
                guard_interval = GUARD_INTERVAL_PLAYING
                effective_position = position

            last_time = self._last_progress_time.get(prov_item_id, 0.0)
            time_since_last = time.time() - last_time
            is_state_change = fully_played or (not is_playing and position == 0)
            if not is_state_change and time_since_last < guard_interval:
                self.logger.debug(
                    "Skipped progress POST for %s: within %ss guard (%.1fs since last)",
                    prov_item_id,
                    guard_interval,
                    time_since_last,
                )
                return

            await self._post_progress(prov_item_id, effective_position)
        except LoginFailed:
            raise
        except Exception as err:
            self.logger.warning("Error syncing progress to Overcast: %s", err)

    async def get_resume_position(
        self,
        item_id: str,
        media_type: MediaType,
    ) -> tuple[bool, int, datetime | None]:
        """
        Return resume state for an episode by scraping its Overcast web page.

        Raises ``NotImplementedError`` (the framework's "I don't know" signal,
        which falls back to MA's own playlog) whenever Overcast data is
        unavailable — no overcast_url cached, network failure, or audio player
        attributes missing from the HTML.

        :param item_id: The overcastId of the episode.
        :param media_type: Media type (always ``PODCAST_EPISODE`` here).
        :return: Tuple of (fully_played, resume_position_ms, timestamp).
        :raises LoginFailed: If Overcast rejects the session cookie.
        """
        overcast_url = await self._get_episode_overcast_url(item_id)
        if not overcast_url:
            raise NotImplementedError

        try:
            async with self.mass.http_session.get(
                overcast_url,
                headers={"Cookie": f"o={self._cookie}"},
                allow_redirects=False,
            ) as resp:
                if self._is_auth_error(resp):
                    raise LoginFailed(ERR_COOKIE_EXPIRED)
                resp.raise_for_status()
                html = await resp.text()
        except LoginFailed:
            raise
        except Exception:
            self.logger.warning("Failed to fetch resume position for %s", item_id)
            raise NotImplementedError from None

        start_time_seconds, sync_version = parse_episode_page(html)
        if start_time_seconds is None:
            raise NotImplementedError

        if sync_version is not None:
            self._sync_versions[item_id] = sync_version

        if start_time_seconds >= FINISHED_SENTINEL:
            return True, 0, None

        return False, start_time_seconds * 1000, None

    async def browse(self, path: str) -> Sequence[BrowseFolder | PodcastEpisode]:
        """
        Resolve a browse path to its child items.

        The tree is::

            <base>                                     -> Playlists, Podcasts
            <base>playlists                            -> playlist folders
            <base>playlists/<title>                    -> playlist episodes
            <base>podcasts                             -> podcast folders
            <base>podcasts/<id>                        -> Unplayed folder + episodes
            <base>podcasts/<id>/unplayed               -> unplayed episodes

        Unknown paths return an empty list.

        :param path: Full browse path (e.g. ``"overcast_test://playlists"``).
        """
        base = f"{self.instance_id}://"
        opml_data = await self._get_opml_data()

        if path == base:
            return browse.browse_root(base, self.domain)

        if not path.startswith(base):
            return []

        subpath_parts = path[len(base) :].split("/")
        subpath = subpath_parts[0] if subpath_parts else ""

        if subpath == BROWSE_PLAYLISTS:
            if len(subpath_parts) == 1:
                return browse.browse_playlists_list(base, self.domain, opml_data)
            playlist_title = unquote(subpath_parts[1])
            episode_lookup = browse.build_episode_lookup(opml_data)
            return browse.browse_playlist_episodes(
                playlist_title, opml_data, episode_lookup, self.domain, self.instance_id
            )

        if subpath == BROWSE_PODCASTS:
            if len(subpath_parts) == 1:
                return browse.browse_podcasts_list(base, self.domain, opml_data)
            if len(subpath_parts) == 2:
                podcast_id = subpath_parts[1]
                unplayed_folder = BrowseFolder(
                    item_id=BROWSE_UNPLAYED,
                    provider=self.domain,
                    path=f"{base}{BROWSE_PODCASTS}/{podcast_id}/{BROWSE_UNPLAYED}",
                    name="Unplayed",
                    translation_key="unplayed",
                )
                episodes = browse.browse_podcast_episodes(
                    podcast_id, opml_data, self.domain, self.instance_id
                )
                return [unplayed_folder, *episodes]
            if len(subpath_parts) >= 3 and subpath_parts[2] == BROWSE_UNPLAYED:
                podcast_id = subpath_parts[1]
                return browse.browse_podcast_episodes(
                    podcast_id, opml_data, self.domain, self.instance_id, unplayed_only=True
                )

        return []

    @property
    def is_streaming_provider(self) -> bool:
        """Return True if the provider is a streaming provider."""
        return True

    async def _validate_auth(self) -> None:
        """
        Probe ``/account`` to confirm the session cookie is still valid.

        :raises LoginFailed: If Overcast rejects the cookie (401/403 or
            redirect to ``/login``).
        """
        async with self.mass.http_session.get(
            OVERCAST_ACCOUNT_URL,
            headers={"Cookie": f"o={self._cookie}"},
            allow_redirects=False,
        ) as resp:
            if self._is_auth_error(resp):
                raise LoginFailed(ERR_COOKIE_EXPIRED)

    @staticmethod
    def _is_auth_error(resp: aiohttp.ClientResponse) -> bool:
        """Return ``True`` if an HTTP response represents an authentication failure."""
        return resp.status in (401, 403) or (
            resp.status in (301, 302, 303, 307, 308)
            and "/login" in str(resp.headers.get("Location", ""))
        )

    @throttle_with_retries
    async def _fetch_opml(self) -> str:
        """
        Fetch the extended OPML export from Overcast.

        :return: OPML XML content as string.
        :raises LoginFailed: If Overcast rejects the session cookie.
        :raises ResourceTemporarilyUnavailable: On transient network/server errors.
        :raises RateLimited: If Overcast rate-limits the request.
        """
        try:
            async with self.mass.http_session.get(
                OVERCAST_OPML_URL,
                headers={"Cookie": f"o={self._cookie}"},
                allow_redirects=False,
            ) as resp:
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After")
                    self.logger.warning(
                        "Overcast rate limited (429), Retry-After: %s",
                        retry_after or "not provided",
                    )
                    raise RateLimited(
                        "Overcast rate limit",
                        backoff_time=parse_retry_after(retry_after),
                    )
                if self._is_auth_error(resp):
                    raise LoginFailed(ERR_COOKIE_EXPIRED)
                # Non-auth 3xx redirects (e.g. canonical URL changes) are not
                # expected for an authenticated API endpoint — surface them as
                # retryable temporary failures rather than letting raise_for_status mask them.
                if resp.status in (301, 302, 303, 307, 308):
                    self.logger.warning(
                        "Unexpected redirect from Overcast OPML endpoint: %s -> %s",
                        resp.status,
                        resp.headers.get("Location", "?"),
                    )
                    raise ResourceTemporarilyUnavailable("Unexpected Overcast OPML redirect")
                if resp.status >= 500:
                    raise ResourceTemporarilyUnavailable(
                        f"Overcast server error ({resp.status})", backoff_time=30
                    )
                resp.raise_for_status()
                return await resp.text()
        except LoginFailed:
            raise
        except aiohttp.ClientError as err:
            raise ResourceTemporarilyUnavailable(
                f"Network error contacting Overcast: {err}"
            ) from err

    async def _get_opml_data(self) -> dict[str, Any]:
        """
        Return parsed OPML data, fetching from Overcast on cache miss.

        :return: Dict with ``"podcasts"`` and ``"playlists"`` keys.
        :raises LoginFailed: If the session cookie is rejected during fetch.
        :raises MusicAssistantError: If no cached data exists and the fetch fails
            transiently (rate-limited or network unreachable).
        """
        cached = await self.mass.cache.get(
            key=CACHE_KEY_OPML,
            provider=self.instance_id,
            category=CACHE_CATEGORY_OPML,
            default=None,
        )
        if cached is not None:
            # Returned without copying — callers in this provider only read.
            # Mutations would propagate to the cache layer, but no caller mutates.
            return cast("dict[str, Any]", cached)

        # Serialize the fetch path: a concurrent caller may have populated the
        # cache while we waited for the lock, so re-check once inside.
        if self._opml_lock is None:
            # handle_async_init normally creates this; guard test paths that
            # bypass init so the lock is still per-instance and reusable.
            self._opml_lock = asyncio.Lock()
        async with self._opml_lock:
            cached = await self.mass.cache.get(
                key=CACHE_KEY_OPML,
                provider=self.instance_id,
                category=CACHE_CATEGORY_OPML,
                default=None,
            )
            if cached is not None:
                return cast("dict[str, Any]", cached)

            self.logger.debug("OPML cache miss, fetching from Overcast")
            try:
                opml_text = await self._fetch_opml()
            except RetriesExhausted as err:
                raise MusicAssistantError(
                    "Could not fetch OPML from Overcast and no cached copy is available. "
                    "Try again shortly — Overcast may be rate-limiting or temporarily unreachable."
                ) from err

            opml_data = parse_opml(opml_text)

            enrichment_raw = await self.mass.cache.get(
                key=CACHE_KEY_RSS_ENRICHMENT,
                provider=self.instance_id,
                category=CACHE_CATEGORY_OPML,
                default=None,
            )
            enrichment_cache = (
                cast("dict[str, dict[str, Any]]", enrichment_raw) if enrichment_raw else {}
            )
            enrichment_updated = False

            for podcast_data in opml_data.get("podcasts", []):
                self._apply_cached_enrichment(podcast_data, enrichment_cache)
                xml_url = podcast_data.get("xml_url")
                if not xml_url:
                    continue

                episodes_missing_enrichment = [
                    ep
                    for ep in podcast_data.get("episodes", [])
                    if ep.get("overcast_id") and not self._has_enrichment(ep)
                ]
                if not episodes_missing_enrichment:
                    self.logger.debug(
                        "Skipping RSS fetch for %s, all episodes already have cached enrichment",
                        podcast_data.get("title", "unknown"),
                    )
                    continue

                try:
                    rss_parsed = await get_podcastparser_dict(
                        session=self.mass.http_session, feed_url=xml_url
                    )
                    enrich_episodes_from_rss(podcast_data, rss_parsed)
                    enrichment_updated |= self._store_cached_enrichment(
                        podcast_data, enrichment_cache
                    )
                except Exception as err:
                    # RSS feeds are third-party and fetched unauthenticated via the
                    # shared session, so LoginFailed cannot originate here (Overcast
                    # auth is never involved). Any failure — 404, invalid RSS, private
                    # feed, transient network issue — leaves OPML-only data intact for
                    # this podcast and lets library sync continue.
                    self.logger.warning(
                        "Failed to fetch RSS for %s, using OPML-only data: %s",
                        podcast_data.get("title", "unknown"),
                        err,
                    )

            if enrichment_updated:
                await self.mass.cache.set(
                    key=CACHE_KEY_RSS_ENRICHMENT,
                    provider=self.instance_id,
                    category=CACHE_CATEGORY_OPML,
                    data=enrichment_cache,
                    expiration=CACHE_TTL_ENRICHMENT,
                )

            await self.mass.cache.set(
                key=CACHE_KEY_OPML,
                provider=self.instance_id,
                category=CACHE_CATEGORY_OPML,
                data=opml_data,
                expiration=CACHE_TTL,
            )

            # Reset per-episode state on fresh OPML fetch to prevent unbounded growth.
            # Sync versions may be stale after refresh; guard interval reset causes
            # at most one extra progress POST per episode.
            self._sync_versions.clear()
            self._last_progress_time.clear()

            return opml_data

    @staticmethod
    def _has_enrichment(episode_data: dict[str, Any]) -> bool:
        """Return ``True`` if an episode already has RSS-derived metadata."""
        return any(key in episode_data for key in ("description", "duration", "cover_url"))

    @staticmethod
    def _apply_cached_enrichment(
        podcast_data: dict[str, Any], enrichment_cache: dict[str, dict[str, Any]]
    ) -> None:
        """Apply cached RSS-derived metadata to OPML episodes in-place."""
        for episode_data in podcast_data.get("episodes", []):
            episode_id = episode_data.get("overcast_id")
            if not episode_id or not (cached := enrichment_cache.get(episode_id)):
                continue
            for key in ("description", "duration", "cover_url"):
                if key in cached:
                    episode_data[key] = cached[key]

    @staticmethod
    def _store_cached_enrichment(
        podcast_data: dict[str, Any], enrichment_cache: dict[str, dict[str, Any]]
    ) -> bool:
        """Store RSS-derived metadata from OPML episodes and return whether cache changed."""
        updated = False
        for episode_data in podcast_data.get("episodes", []):
            episode_id = episode_data.get("overcast_id")
            if not episode_id:
                continue
            enrichment = {
                key: episode_data[key]
                for key in ("description", "duration", "cover_url")
                if key in episode_data
            }
            if enrichment and enrichment_cache.get(episode_id) != enrichment:
                enrichment_cache[episode_id] = enrichment
                updated = True
        return updated

    async def _find_episode_in_opml(self, episode_id: str) -> dict[str, Any] | None:
        """
        Locate an episode dict in the cached OPML by overcastId.

        :param episode_id: The overcastId of the episode.
        :return: The episode dict if found, otherwise ``None``. The dict is
            shared with the cache; callers must not mutate it.
        """
        opml_data = await self._get_opml_data()
        for podcast_data in opml_data.get("podcasts", []):
            for episode_data in podcast_data.get("episodes", []):
                if episode_data.get("overcast_id") == episode_id:
                    return cast("dict[str, Any]", episode_data)
        return None

    async def _post_progress(self, episode_id: str, position_seconds: int) -> None:
        """
        POST playback progress to Overcast's set_progress endpoint.

        :param episode_id: The overcastId of the episode (must be numeric).
        :param position_seconds: Playback position in seconds, or
            ``FINISHED_SENTINEL`` to mark fully played.
        :raises LoginFailed: If Overcast rejects the session cookie.
        """
        # Defense in depth: overcastId comes from external OPML XML. Reject
        # anything non-numeric before letting it become part of the request URL.
        if not episode_id.isdigit():
            self.logger.warning(
                "Refusing to POST progress for non-numeric episode_id %r", episode_id
            )
            return

        sync_version = self._sync_versions.get(episode_id, 0)
        url = f"{OVERCAST_SET_PROGRESS_URL}/{episode_id}"

        try:
            async with self.mass.http_session.post(
                url,
                # speed=0 tells Overcast "no speed override" (preserves user's app setting)
                data={
                    "p": str(position_seconds),
                    "speed": "0",
                    "v": str(sync_version),
                },
                headers={"Cookie": f"o={self._cookie}"},
                allow_redirects=False,
            ) as resp:
                if self._is_auth_error(resp):
                    raise LoginFailed(ERR_COOKIE_EXPIRED)
                # Update the rate-limit timestamp on any HTTP response (success
                # OR error). Without this, a server returning persistent 5xx
                # would let every subsequent on_played call bypass the guard
                # and hammer the broken endpoint at the playback heartbeat rate.
                # Network/transport errors raise out of the `async with` and
                # bypass this update — those are transient and worth retrying.
                self._last_progress_time[episode_id] = time.time()
                if resp.status == 200:
                    response_text = (await resp.text()).strip()
                    try:
                        self._sync_versions[episode_id] = int(response_text)
                        self.logger.debug(
                            "Updated sync version for %s to %s", episode_id, response_text
                        )
                    except ValueError, TypeError:
                        self.logger.debug(
                            "Could not parse sync version from response, keeping v=%s",
                            sync_version,
                        )
                    self.logger.debug(
                        "Posted progress to Overcast: %s @ %ss", episode_id, position_seconds
                    )
                else:
                    self.logger.warning(
                        "Failed to post progress: %s (status %s)", episode_id, resp.status
                    )
        except LoginFailed:
            raise
        except Exception as err:
            self.logger.warning("Failed to sync progress to Overcast: %s", err)

    async def _get_episode_overcast_url(self, episode_id: str) -> str | None:
        """Return the cached ``overcast_url`` for an episode, or ``None``."""
        episode_data = await self._find_episode_in_opml(episode_id)
        if episode_data is None:
            return None
        return episode_data.get("overcast_url") or None
