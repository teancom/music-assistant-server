"""Constants for the Overcast provider."""

from music_assistant_models.enums import ProviderFeature

# Configuration keys
CONF_COOKIE = "session_cookie"

# Overcast URLs
OVERCAST_BASE_URL = "https://overcast.fm"
OVERCAST_ACCOUNT_URL = f"{OVERCAST_BASE_URL}/account"
OVERCAST_OPML_URL = f"{OVERCAST_BASE_URL}/account/export_opml/extended"

# Cache settings
CACHE_CATEGORY_OPML = 0
CACHE_KEY_OPML = "opml_data"
CACHE_TTL = 60 * 60 * 12  # 12 hours - matches default library sync interval
CACHE_KEY_RSS_ENRICHMENT = "rss_enrichment"
CACHE_TTL_ENRICHMENT = 60 * 60 * 24 * 30  # 30 days

# Playback sync
OVERCAST_SET_PROGRESS_URL = f"{OVERCAST_BASE_URL}/podcasts/set_progress"
FINISHED_SENTINEL = 2147483647  # 0x7FFFFFFF — marks episode as fully played in Overcast

# Guard intervals for on_played write-back (seconds)
GUARD_INTERVAL_PLAYING = 60
GUARD_INTERVAL_ACTION = 1

# Browse paths
BROWSE_PLAYLISTS = "playlists"
BROWSE_PODCASTS = "podcasts"
BROWSE_UNPLAYED = "unplayed"

# Allowed image file extensions for cover art URLs.
# Some RSS feeds advertise their feed URL or an HTML page as cover_url; reject
# anything that isn't a known image extension to prevent PIL errors downstream.
# Extensionless URLs are accepted (the upstream image fetcher validates Content-Type).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif")

# Shared error message — emitted whenever Overcast rejects the session cookie
# (401/403 or redirect to /login). Surfaced to the user via LoginFailed so the
# provider config UI prompts for a new cookie.
ERR_COOKIE_EXPIRED = "Overcast session cookie has expired. Please update the cookie value."

# Supported features
SUPPORTED_FEATURES = {
    ProviderFeature.LIBRARY_PODCASTS,
    ProviderFeature.BROWSE,
}
