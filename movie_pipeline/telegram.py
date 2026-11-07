"""Send a completion reminder; never upload media or expose bot credentials."""
from __future__ import annotations

import os
from html import escape
from pathlib import Path

import requests


def notify() -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram reminder skipped: configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID repository secrets.")
        return False
    title_file = Path("output/recommended-title.txt")
    title = title_file.read_text(encoding="utf-8").strip() if title_file.exists() else os.getenv("VIDEO_TITLE", "Your video")
    title = " ".join(title.split())[:300]
    run_url = f"{os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    gap = "\n" * 6  # Six line breaks leave five empty lines between sections.
    message = gap.join([
        "✅ <b>Your video is ready!</b>",
        f"<b>Title:</b> {escape(title)}",
        f'<b>Download:</b> <a href="{escape(run_url, quote=True)}">GitHub run link</a>',
    ]) + "\n\nOpen Artifacts to download. Available for 1 day."
    try:
        response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }, timeout=20)
        if response.status_code != 200:
            print(f"Telegram reminder was not delivered (HTTP {response.status_code}). Check bot/chat secrets and start a chat with the bot. The video package is still available in Artifacts.")
            return False
        data = response.json()
        if not data.get("ok"):
            print("Telegram did not accept the reminder. The video package remains in Artifacts.")
            return False
    except (requests.RequestException, ValueError):
        # Request exceptions contain the URL, which embeds the token. Never print them.
        print("Telegram reminder could not be confirmed due to a network/response error. The video package remains in Artifacts.")
        return False
    print("Telegram completion reminder delivered.")
    message_id = data.get("result", {}).get("message_id")
    if not isinstance(message_id, int):
        print("Reminder sent but pinning skipped: Telegram returned no message ID.")
        return True
    try:
        pinned = requests.post(f"https://api.telegram.org/bot{token}/pinChatMessage", json={
            "chat_id": chat_id, "message_id": message_id, "disable_notification": False,
        }, timeout=20)
        if pinned.status_code == 200 and pinned.json().get("ok"):
            print("Telegram completion reminder pinned.")
        else:
            print("Reminder sent but could not be pinned. Give the bot Edit Messages rights in a channel, or Pin Messages rights in a group. The video package is unaffected.")
    except (requests.RequestException, ValueError):
        print("Reminder sent; pinning could not be confirmed. The video package is unaffected.")
    return True


if __name__ == "__main__":
    notify()
