"""
Market Bot v26 — Watchdog Service
==================================
Wraps market_bot_v26.py with:
  - Auto-restart on crash (30s delay)
  - Health monitoring (log freshness, error storms)
  - Telegram alerts on crash/restart
  - Clean shutdown via Ctrl+C (forwarded to bot)
  - Crash counter with backoff (prevents rapid restart loops)

Usage:
    python run_service.py              # Start with watchdog
    python market_bot_v26.py           # Direct launch (no watchdog)
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path
from datetime import datetime, timedelta

# --- Configuration ---
BOT_SCRIPT = "market_bot_v26.py"
BOT_DIR = Path(__file__).parent
RESTART_DELAY = 30          # seconds between crash and restart
MAX_RAPID_CRASHES = 5       # if this many crashes in CRASH_WINDOW, back off
CRASH_WINDOW = 300          # seconds (5 min)
BACKOFF_DELAY = 300         # seconds to wait after too many rapid crashes
HEALTH_CHECK_INTERVAL = 900 # seconds (15 min)
LOG_STALE_THRESHOLD = 2700  # seconds (45 min) — alert if log not written


def _send_telegram(text: str):
    """Send a Telegram alert using the bot's own credentials."""
    try:
        env_path = BOT_DIR / ".env"
        if not env_path.exists():
            return
        token, chat_id = None, None
        for line in env_path.read_text().splitlines():
            if line.startswith("TELEGRAM_TOKEN="):
                token = line.split("=", 1)[1].strip()
            elif line.startswith("TELEGRAM_CHAT_ID="):
                chat_id = line.split("=", 1)[1].strip()
        if not token or not chat_id or len(token) < 20:
            return

        import urllib.request
        import urllib.parse
        import json
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # Don't let alert failure break the watchdog


def _check_log_health() -> list:
    """Check today's log file for staleness and error storms."""
    issues = []
    now = datetime.now()
    log_dir = BOT_DIR / "logs"
    log_file = log_dir / f"bot_{now:%Y%m%d}.log"

    if not log_file.exists():
        if now.hour >= 10:
            issues.append("No log file for today (after 10 AM)")
        return issues

    # Staleness check
    age = time.time() - log_file.stat().st_mtime
    if age > LOG_STALE_THRESHOLD:
        issues.append(f"Log stale: last written {int(age/60)} min ago")

    # Error storm check (last 20 lines)
    try:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        error_count = sum(1 for l in lines if "[ERROR]" in l)
        if error_count >= 5:
            issues.append(f"Error storm: {error_count} errors in last 20 lines")
    except Exception:
        pass

    return issues


def main():
    print("=" * 60)
    print("  Market Bot v26 — Watchdog Service")
    print("=" * 60)
    print(f"  Bot script:    {BOT_SCRIPT}")
    print(f"  Working dir:   {BOT_DIR}")
    print(f"  Restart delay: {RESTART_DELAY}s")
    print(f"  Health check:  every {HEALTH_CHECK_INTERVAL//60} min")
    print("=" * 60)
    print()

    crash_times = []
    bot_process = None
    shutting_down = False

    def _shutdown(signum=None, frame=None):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        print("\n[Watchdog] Shutting down...")
        if bot_process and bot_process.poll() is None:
            # Send Ctrl+C to the bot's process group
            try:
                os.kill(bot_process.pid, signal.CTRL_C_EVENT)
                print("[Watchdog] Sent Ctrl+C to bot, waiting up to 30s for cleanup...")
                bot_process.wait(timeout=30)
                print("[Watchdog] Bot exited cleanly.")
            except subprocess.TimeoutExpired:
                print("[Watchdog] Bot didn't stop in 30s, killing...")
                bot_process.kill()
            except Exception:
                bot_process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGBREAK, _shutdown)

    last_health_check = time.time()

    while not shutting_down:
        # Launch the bot
        print(f"[Watchdog] Starting {BOT_SCRIPT}...")
        start_time = time.time()

        try:
            bot_process = subprocess.Popen(
                [sys.executable, BOT_SCRIPT, "--headless"],
                cwd=str(BOT_DIR),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        except Exception as e:
            print(f"[Watchdog] Failed to start bot: {e}")
            _send_telegram(f"*WATCHDOG ERROR*\nFailed to start bot: `{e}`")
            time.sleep(RESTART_DELAY)
            continue

        print(f"[Watchdog] Bot running (PID {bot_process.pid})")

        # Monitor loop: check if bot is alive + periodic health checks
        while not shutting_down:
            try:
                exit_code = bot_process.wait(timeout=HEALTH_CHECK_INTERVAL)

                # Bot exited
                runtime = int(time.time() - start_time)
                if shutting_down:
                    break

                print(f"\n[Watchdog] Bot exited with code {exit_code} after {runtime}s")

                # Track crash for backoff
                now = time.time()
                crash_times.append(now)
                crash_times = [t for t in crash_times if now - t < CRASH_WINDOW]

                if exit_code == 0:
                    # Clean exit (e.g. daily loss limit sleep) — restart normally
                    _send_telegram(
                        f"*WATCHDOG: Bot exited cleanly*\n"
                        f"Runtime: {runtime}s | Restarting in {RESTART_DELAY}s..."
                    )
                    print(f"[Watchdog] Clean exit. Restarting in {RESTART_DELAY}s...")
                    time.sleep(RESTART_DELAY)
                elif len(crash_times) >= MAX_RAPID_CRASHES:
                    # Too many crashes in short window — back off
                    _send_telegram(
                        f"*WATCHDOG: {len(crash_times)} crashes in {CRASH_WINDOW//60} min*\n"
                        f"Backing off for {BACKOFF_DELAY//60} min before retry..."
                    )
                    print(f"[Watchdog] {len(crash_times)} rapid crashes! Backing off {BACKOFF_DELAY}s...")
                    time.sleep(BACKOFF_DELAY)
                    crash_times.clear()
                else:
                    # Normal crash — restart after delay
                    _send_telegram(
                        f"*WATCHDOG: Bot crashed*\n"
                        f"Exit code: {exit_code} | Runtime: {runtime}s\n"
                        f"Restarting in {RESTART_DELAY}s..."
                    )
                    print(f"[Watchdog] Restarting in {RESTART_DELAY}s...")
                    time.sleep(RESTART_DELAY)
                break  # Exit inner loop to restart

            except subprocess.TimeoutExpired:
                # Bot still running — do health check
                now = time.time()
                if now - last_health_check >= HEALTH_CHECK_INTERVAL:
                    issues = _check_log_health()
                    if issues:
                        print(f"[Watchdog] Health issues: {issues}")
                    last_health_check = now


if __name__ == "__main__":
    main()
