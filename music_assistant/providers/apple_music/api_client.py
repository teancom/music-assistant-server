"""Apple Music API client."""

from __future__ import annotations

import functools
import sys
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Concatenate

from aiohttp import ClientConnectionError, ClientPayloadError
from music_assistant_models.enums import MediaType
from music_assistant_models.errors import (
    MediaNotFoundError,
    ResourceTemporarilyUnavailable,
)

from music_assistant.helpers.json import json_loads
from music_assistant.helpers.throttle_retry import ThrottlerManager, throttle_with_retries

from .helpers.utils import is_library_id, translate_media_type_to_apple_type

if TYPE_CHECKING:
    from .provider import AppleMusicProvider

_APPLE_API_BASE = "https://api.music.apple.com/v1"

# How many times to re-fetch a single page that came back truncated mid-pagination
# before giving up and surfacing the failure.
_PAGE_TRUNCATION_RETRIES = 3


def _retry_transient_transport_errors[ClientT, **P, R](
    func: Callable[Concatenate[ClientT, P], Awaitable[R]],
) -> Callable[Concatenate[ClientT, P], Awaitable[R]]:
    """
    Convert transient aiohttp transport errors into the retryable error type.

    A dropped connection or truncated body raises an ``aiohttp.ClientError`` (or
    ``TimeoutError``) rather than an HTTP status, so it would otherwise bypass the
    status-based retry handling. ``ClientResponseError`` (raised by
    ``raise_for_status`` for genuine 4xx/5xx) is deliberately not caught here.
    """

    @functools.wraps(func)
    async def wrapper(self: ClientT, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await func(self, *args, **kwargs)
        except (ClientConnectionError, ClientPayloadError, TimeoutError) as err:
            raise ResourceTemporarilyUnavailable(
                f"Transient transport error calling Apple Music: {err}"
            ) from err

    return wrapper


# --- DEBUG (throwaway, branch: debug/apple-music-memory) memory instrumentation ---
_DEBUG_MEM = True


def _proc_mem_mb(field: str) -> float:
    """Read a /proc/self/status memory field (e.g. 'VmRSS:') in MB; -1 if unavailable."""
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith(field):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except OSError:
        pass
    return -1.0


def _deep_size(obj: object, _seen: set[int] | None = None) -> int:
    """Recursively size a JSON-like object graph in bytes, deduping shared refs by id."""
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return 0
    _seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_size(k, _seen) + _deep_size(v, _seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += _deep_size(item, _seen)
    return size


class AppleMusicAPIClient:
    """Handles all HTTP communication with the Apple Music API."""

    # Tuning knobs under observation: watch dev logs for 429s (raise period) or
    # "Apple Music API Timeout" 504s (lower the page limit in get_all_items).
    # period=0.25 -> 4 req/s.
    throttler = ThrottlerManager(rate_limit=1, period=0.25, initial_backoff=15)

    def __init__(self, provider: AppleMusicProvider) -> None:
        """Initialize the API client."""
        self.provider = provider
        self.logger = provider.logger

    @property
    def _headers(self) -> dict[str, str]:
        """Return standard auth headers."""
        return {
            "Authorization": f"Bearer {self.provider._music_app_token}",
            "Music-User-Token": self.provider._music_user_token,
        }

    @throttle_with_retries
    @_retry_transient_transport_errors
    async def get_data(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """GET data from the Apple Music API."""
        url = f"{_APPLE_API_BASE}/{endpoint}"
        async with (
            self.provider.mass.http_session.get(
                url, headers=self._headers, params=kwargs, ssl=True, timeout=120
            ) as response,
        ):
            if response.status == 404 and "limit" in kwargs and "offset" in kwargs:
                return {}
            if response.status == 404:
                raise MediaNotFoundError(f"{endpoint} not found")
            if response.status == 504:
                self.provider.logger.debug(
                    "Apple Music API Timeout: url=%s, params=%s, response_headers=%s",
                    url,
                    kwargs,
                    response.headers,
                )
                raise ResourceTemporarilyUnavailable("Apple Music API Timeout")
            if response.status == 429:
                self.provider.logger.debug(
                    "Apple Music Rate Limiter. Headers: %s", response.headers
                )
                raise ResourceTemporarilyUnavailable("Apple Music Rate Limiter")
            if response.status == 500:
                # Apple 500s are typically transient (especially under load); retry rather
                # than aborting the whole sync on a single hiccup.
                raise ResourceTemporarilyUnavailable(
                    "Unexpected server error when calling Apple Music"
                )
            response.raise_for_status()
            return await response.json(loads=json_loads)

    @throttle_with_retries
    async def delete_data(self, endpoint: str, data: Any = None, **kwargs: Any) -> None:
        """DELETE data from the Apple Music API."""
        url = f"{_APPLE_API_BASE}/{endpoint}"
        async with (
            self.provider.mass.http_session.delete(
                url, headers=self._headers, params=kwargs, json=data, ssl=True, timeout=120
            ) as response,
        ):
            if response.status == 404:
                raise MediaNotFoundError(f"{endpoint} not found")
            if response.status == 429:
                self.provider.logger.debug(
                    "Apple Music Rate Limiter. Headers: %s", response.headers
                )
                raise ResourceTemporarilyUnavailable("Apple Music Rate Limiter")
            response.raise_for_status()

    @throttle_with_retries
    async def put_data(self, endpoint: str, data: Any = None, **kwargs: Any) -> dict[str, Any]:
        """PUT data to the Apple Music API."""
        url = f"{_APPLE_API_BASE}/{endpoint}"
        async with (
            self.provider.mass.http_session.put(
                url, headers=self._headers, params=kwargs, json=data, ssl=True, timeout=120
            ) as response,
        ):
            if response.status == 404:
                raise MediaNotFoundError(f"{endpoint} not found")
            if response.status == 429:
                self.provider.logger.debug(
                    "Apple Music Rate Limiter. Headers: %s", response.headers
                )
                raise ResourceTemporarilyUnavailable("Apple Music Rate Limiter")
            response.raise_for_status()
            if response.content_length:
                return await response.json(loads=json_loads)
            return {}

    @throttle_with_retries
    async def post_data(self, endpoint: str, data: Any = None, **kwargs: Any) -> dict[str, Any]:
        """POST data to the Apple Music API."""
        url = f"{_APPLE_API_BASE}/{endpoint}"
        async with (
            self.provider.mass.http_session.post(
                url, headers=self._headers, params=kwargs, json=data, ssl=True, timeout=120
            ) as response,
        ):
            if response.status == 404:
                raise MediaNotFoundError(f"{endpoint} not found")
            if response.status == 429:
                self.provider.logger.debug(
                    "Apple Music Rate Limiter. Headers: %s", response.headers
                )
                raise ResourceTemporarilyUnavailable("Apple Music Rate Limiter")
            response.raise_for_status()
            return await response.json(loads=json_loads)

    async def get_all_items(self, endpoint: str, key: str = "data", **kwargs: Any) -> list[dict]:
        """Get all items from a paged list."""
        limit = 50  # known-safe page size; larger pages 504 at deep offsets on heavy includes
        offset = 0
        all_items: list[dict] = []
        dbg_next = 1500
        if _DEBUG_MEM:
            self.logger.warning("[MEMDEBUG] start %s rss=%.1fMB", endpoint, _proc_mem_mb("VmRSS:"))
        while True:
            kwargs["limit"] = limit
            kwargs["offset"] = offset
            result = await self._get_page(endpoint, key, offset, kwargs)
            if key not in result:
                # only reachable on the first page (offset 0): a genuinely empty
                # collection or a 404; either way there is nothing to paginate.
                break
            all_items += result[key]
            if _DEBUG_MEM and len(all_items) >= dbg_next:
                window = all_items[-150:]
                self.logger.warning(
                    "[MEMDEBUG] %s listed=%d rss=%.1fMB hwm=%.1fMB deepavg/track=%.0fB",
                    endpoint,
                    len(all_items),
                    _proc_mem_mb("VmRSS:"),
                    _proc_mem_mb("VmHWM:"),
                    _deep_size(window) / len(window),
                )
                dbg_next += 1500
            if not result.get("next"):
                break
            offset += limit
        if _DEBUG_MEM:
            self.logger.warning(
                "[MEMDEBUG] done %s total=%d rss=%.1fMB hwm=%.1fMB",
                endpoint,
                len(all_items),
                _proc_mem_mb("VmRSS:"),
                _proc_mem_mb("VmHWM:"),
            )
        return all_items

    async def get_user_storefront(self) -> str:
        """Return the user's storefront identifier."""
        locale = self.provider.mass.metadata.locale.replace("_", "-")
        language = locale.split("-")[0]
        result = await self.get_data("me/storefront", l=language)
        return result["data"][0]["id"]

    async def get_ratings(self, item_ids: list[str], media_type: MediaType) -> dict[str, bool]:
        """Return a mapping of item_id → is_favourite for a list of IDs."""
        if media_type == MediaType.ARTIST:
            raise NotImplementedError(
                "Ratings are not available for artist in the Apple Music API."
            )
        if not item_ids:
            return {}
        apple_type = translate_media_type_to_apple_type(media_type)
        endpoint = apple_type if not is_library_id(item_ids[0]) else f"library-{apple_type}"
        max_ids_per_request = 200
        results: dict[str, bool] = {}
        for i in range(0, len(item_ids), max_ids_per_request):
            batch_ids = item_ids[i : i + max_ids_per_request]
            response = await self.get_data(
                f"me/ratings/{endpoint}",
                ids=",".join(batch_ids),
            )
            results.update(
                {
                    item["id"]: bool(item["attributes"].get("value", False) == 1)
                    for item in response.get("data", [])
                }
            )
        return results

    async def _get_page(
        self, endpoint: str, key: str, offset: int, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Fetch a single page of a paged listing, recovering from transient truncation.

        Apple returns ``{}`` (HTTP 404) for a paged request that yields no payload. On the
        first page (offset 0) that legitimately means an empty collection. Mid-pagination
        (offset > 0, reached only after a prior page promised more via ``next``) it means a
        transient truncation; accepting it would drop still-present items and trigger
        spurious deletions, so re-fetch the same page a bounded number of times before
        surfacing the failure.
        """
        result = await self.get_data(endpoint, **kwargs)
        if key in result or offset == 0:
            return result
        for _ in range(_PAGE_TRUNCATION_RETRIES):
            result = await self.get_data(endpoint, **kwargs)
            if key in result:
                return result
        raise ResourceTemporarilyUnavailable(
            f"Incomplete paged listing for {endpoint} at offset {offset}"
        )
