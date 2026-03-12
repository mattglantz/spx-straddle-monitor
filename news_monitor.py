"""
News Monitor — RSS feed poller with Telegram alerts.

Polls financial RSS feeds on a configurable interval, deduplicates
by link, and sends formatted Telegram messages for new articles
matching optional keyword filters.

Stores seen-article history in a local SQLite DB so it survives restarts.
"""

import os
import sys
import time
import sqlite3
import hashlib
from urllib.parse import urlsplit
import logging
import logging.handlers
import signal
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

POLL_INTERVAL_SECONDS = int(os.getenv("NEWS_POLL_INTERVAL", "30"))  # 30 sec default
MAX_ARTICLE_AGE_HOURS = int(os.getenv("NEWS_MAX_AGE_HOURS", "24"))
DB_PATH = Path(__file__).parent / "news_monitor.db"

# --- RSS Feeds -----------------------------------------------------------------
# Add/remove feeds here. Each entry is (name, url).
FEEDS: list[tuple[str, str]] = [
    ("Reuters Markets",    "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best"),
    ("CNBC Top News",      "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("CNBC Economy",       "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
    ("MarketWatch Top",    "http://feeds.marketwatch.com/marketwatch/topstories/"),
    ("MarketWatch Markets","http://feeds.marketwatch.com/marketwatch/marketpulse/"),
    # ("Yahoo Finance",      "https://finance.yahoo.com/news/rssindex"),  # too much clickbait
    ("Fed Reserve News",   "https://www.federalreserve.gov/feeds/press_all.xml"),
    ("Forex Factory",      "https://www.forexfactory.com/rss.php"),
]

# Keywords to match (case-insensitive). Empty list = send ALL articles.
# Tight filter: only headlines that actually move ES/SPX.
KEYWORDS: list[str] = [
    # People who move markets (matched against title only)
    "Powell says", "Powell warns", "Bessent says",
    # Policy / geopolitics
    "new tariff", "tariff hike", "tariff increase", "tariff threat", "tariff delay", "tariff pause",
    "sanctions", "trade war", "executive order",
    "rate cut", "rate hike", "FOMC", "fed holds", "fed raises", "fed cuts",
    # Key data releases (exact terms)
    "CPI ", "PPI ", "PCE ", "nonfarm payroll", "jobs report",
    "GDP ", "JOLTS", "retail sales",
    # Market dislocations (index-level only)
    "circuit breaker", "limit down", "limit up",
    "market selloff", "stock market crash",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("news_monitor")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")

_fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / "news_monitor.log", maxBytes=2_000_000, backupCount=3)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
logger.addHandler(_sh)

# ---------------------------------------------------------------------------
# HTTP session with retries
# ---------------------------------------------------------------------------
_session = requests.Session()
_adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                                         status_forcelist=[429, 500, 502, 503, 504]))
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

# ---------------------------------------------------------------------------
# SQLite history (dedup across restarts)
# ---------------------------------------------------------------------------
def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_articles (
            link_hash  TEXT PRIMARY KEY,
            title_hash TEXT,
            title      TEXT,
            link       TEXT,
            feed       TEXT,
            seen_at    TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_seen_at ON seen_articles(seen_at)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_title_hash ON seen_articles(title_hash)
    """)
    # Migration: add title_hash column if missing (existing DBs)
    try:
        conn.execute("SELECT title_hash FROM seen_articles LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE seen_articles ADD COLUMN title_hash TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_title_hash ON seen_articles(title_hash)")
    conn.commit()
    return conn


def _normalize_link(link: str) -> str:
    """Strip query params so the same article from different feeds deduplicates."""
    parts = urlsplit(link)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def _title_hash(title: str) -> str:
    """Normalize title for dedup: lowercase, strip punctuation/whitespace."""
    import re
    cleaned = re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()
    return hashlib.sha256(cleaned.encode()).hexdigest()


def _is_seen(conn: sqlite3.Connection, link: str, title: str = "") -> bool:
    h = hashlib.sha256(_normalize_link(link).encode()).hexdigest()
    row = conn.execute("SELECT 1 FROM seen_articles WHERE link_hash = ?", (h,)).fetchone()
    if row:
        return True
    # Also check by title to catch same story from different URLs
    if title:
        th = _title_hash(title)
        row = conn.execute("SELECT 1 FROM seen_articles WHERE title_hash = ?", (th,)).fetchone()
        if row:
            return True
    return False


def _mark_seen(conn: sqlite3.Connection, title: str, link: str, feed: str):
    h = hashlib.sha256(_normalize_link(link).encode()).hexdigest()
    th = _title_hash(title) if title else None
    now = datetime.now(ET).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO seen_articles (link_hash, title_hash, title, link, feed, seen_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (h, th, title, link, feed, now),
    )
    conn.commit()


def _prune_old(conn: sqlite3.Connection, max_age_hours: int = 72):
    cutoff = (datetime.now(ET) - timedelta(hours=max_age_hours)).isoformat()
    conn.execute("DELETE FROM seen_articles WHERE seen_at < ?", (cutoff,))
    conn.commit()

# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------
def _matches_keywords(title: str, summary: str) -> bool:
    if not KEYWORDS:
        return True
    # Match against title only — summaries contain too much noise
    text = title.lower()
    # Trump: direct quotes OR market-moving policy actions
    if "trump" in text:
        # Direct quote
        if any(q in title for q in ['"', '\u2018', '\u2019', '\u201c', '\u201d', ':']):
            return True
        # Market-moving Trump actions (specific policy terms only)
        trump_movers = ["tariff", "trade war", "china", "canada", "mexico", "eu ", "europe",
                        "executive order",
                        "sanction", "tax cut", "tax hike", "debt ceiling",
                        "ban", "restrict"]
        if any(kw in text for kw in trump_movers):
            return True
    return any(kw.lower() in text for kw in KEYWORDS)

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def send_telegram(text: str, retries: int = 2) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing — skipping send")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                return True
            logger.error("Telegram send failed (%s): %s", r.status_code, r.text[:200])
        except Exception as e:
            logger.warning("Telegram send attempt %d error: %s", attempt, e)
            if attempt < retries:
                time.sleep(2)
                continue
            logger.error("Telegram send failed after %d attempts", retries)
    return False


def test_telegram() -> bool:
    now = datetime.now(ET).strftime("%I:%M %p ET")
    msg = f"\U0001f4f0 <b>News Monitor Online</b>\n{now}\nMonitoring {len(FEEDS)} feeds"
    ok = send_telegram(msg)
    if ok:
        logger.info("Telegram test message sent successfully")
    else:
        logger.error("Telegram test FAILED")
    return ok

# ---------------------------------------------------------------------------
# Feed polling
# ---------------------------------------------------------------------------
def seed_feeds(conn: sqlite3.Connection):
    """First-run: mark all existing articles as seen without sending alerts."""
    total = 0
    for feed_name, feed_url in FEEDS:
        try:
            d = feedparser.parse(feed_url)
            for entry in d.entries:
                link = getattr(entry, "link", "") or ""
                title = getattr(entry, "title", "") or ""
                if link and not _is_seen(conn, link, title):
                    _mark_seen(conn, title, link, feed_name)
                    total += 1
        except Exception as e:
            logger.warning("Seed error [%s]: %s", feed_name, e)
    logger.info("Seeded %d existing articles (no alerts sent)", total)


def poll_feeds(conn: sqlite3.Connection) -> int:
    new_count = 0
    for feed_name, feed_url in FEEDS:
        try:
            d = feedparser.parse(feed_url)
            if d.bozo and not d.entries:
                logger.warning("Feed error [%s]: %s", feed_name, d.bozo_exception)
                continue

            for entry in d.entries:
                link = getattr(entry, "link", "") or ""
                title = getattr(entry, "title", "No title") or "No title"
                summary = getattr(entry, "summary", "") or ""

                if not link or _is_seen(conn, link, title):
                    continue

                if not _matches_keywords(title, summary):
                    _mark_seen(conn, title, link, feed_name)
                    continue

                # Build and send alert
                clean_title = title.replace("<", "&lt;").replace(">", "&gt;")
                msg = (
                    f"\U0001f4f0 <b>{feed_name}</b>\n"
                    f"{clean_title}\n"
                    f"<a href=\"{link}\">Read more</a>"
                )
                send_telegram(msg)
                _mark_seen(conn, title, link, feed_name)
                new_count += 1
                time.sleep(1.5)  # rate-limit Telegram sends

        except Exception as e:
            logger.error("Error polling [%s]: %s", feed_name, e)

    return new_count

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
_running = True

def _handle_signal(sig, frame):
    global _running
    logger.info("Shutdown signal received")
    _running = False

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def main():
    global _running

    logger.info("=" * 50)
    logger.info("News Monitor starting")
    logger.info("Feeds: %d | Poll interval: %ds | Keywords: %d",
                len(FEEDS), POLL_INTERVAL_SECONDS, len(KEYWORDS))
    logger.info("=" * 50)

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set in .env")
        sys.exit(1)

    # Test Telegram on startup
    if not test_telegram():
        logger.error("Cannot reach Telegram — check token/chat_id")
        sys.exit(1)

    conn = _init_db()
    _prune_old(conn)

    # First run: seed existing articles so we only alert on NEW ones
    row = conn.execute("SELECT COUNT(*) FROM seen_articles").fetchone()
    if row[0] == 0:
        logger.info("First run detected — seeding existing articles...")
        seed_feeds(conn)

    cycle = 0
    while _running:
        cycle += 1
        logger.info("Poll cycle %d ...", cycle)
        try:
            n = poll_feeds(conn)
            if n:
                logger.info("  -> %d new article(s) sent", n)
            else:
                logger.info("  -> No new matching articles")
        except Exception as e:
            logger.error("Poll cycle error: %s", e)

        # Prune old entries every 50 cycles
        if cycle % 50 == 0:
            _prune_old(conn)

        # Interruptible sleep
        for _ in range(POLL_INTERVAL_SECONDS):
            if not _running:
                break
            time.sleep(1)

    conn.close()
    logger.info("News Monitor stopped")
    send_telegram("\U0001f6d1 <b>News Monitor Offline</b>")


if __name__ == "__main__":
    main()
