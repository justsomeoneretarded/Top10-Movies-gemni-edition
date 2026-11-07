"""Shared yt-dlp access settings: cookies, PO tokens and an optional fallback route.

Order of attempts for YouTube footage:
1. direct runner network (with the private cookies secret when configured;
   the bgutil PO-token plugin is used automatically when its server runs);
2. ``YTDLP_FALLBACK_PROXY`` (a Cloudflare WARP SOCKS route prepared by the
   workflow), used only after the direct route meets a sign-in/bot check.

Once the fallback has been needed, later downloads in the same run go straight
to it so each title does not repeat the blocked attempt.
"""
from __future__ import annotations

import os

_state = {"prefer_fallback": False}


def fallback_proxy() -> str:
    return os.getenv("YTDLP_FALLBACK_PROXY", "").strip()


def routes() -> list[str]:
    """Network routes to try, best first."""
    if not fallback_proxy():
        return ["direct"]
    return ["fallback", "direct"] if _state["prefer_fallback"] else ["direct", "fallback"]


def options(base: dict, route: str = "direct") -> dict:
    """yt-dlp options for a route; never mutates ``base``."""
    result = dict(base)
    cookies = os.getenv("YOUTUBE_COOKIE_FILE")
    if cookies:
        result["cookiefile"] = cookies
    if route == "fallback" and fallback_proxy():
        result["proxy"] = fallback_proxy()
    return result


def record_route_success(route: str):
    if route == "fallback":
        if not _state["prefer_fallback"]:
            print("Trailer downloads now use the fallback network route for the rest of this run.", flush=True)
        _state["prefer_fallback"] = True


def reset():
    _state["prefer_fallback"] = False
