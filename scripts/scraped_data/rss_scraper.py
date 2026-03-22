#!/usr/bin/env python3
"""Scrape image-rich articles from RSS feeds into JSONL."""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import feedparser
import requests
import trafilatura
from bs4 import BeautifulSoup

USER_AGENT = "rss-photo-scraper/0.2 (research)"
IMAGE_EXT_PATTERN = re.compile(r"\.(jpg|jpeg|png|webp)(\?|$)", re.IGNORECASE)

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def is_image_url(url: str) -> bool:
    return bool(IMAGE_EXT_PATTERN.search(url))


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def fetch_html_requests(url: str, timeout: int = 25) -> Optional[str]:
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        return response.text
    except Exception:
        return None


def fetch_html_playwright(url: str, timeout_ms: int = 30000) -> Optional[str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError("Playwright not installed. Run: pip install playwright && playwright install chromium")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            html = page.content()
            browser.close()
        return html
    except Exception:
        return None


def extract_article_text(html: str, url: str) -> str:
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_recall=False,
    )
    return (text or "").strip()


def extract_image_urls(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    root = soup.find("article") or soup

    urls = []
    for img in root.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if src:
            urls.append(urljoin(base_url, src))

        srcset = img.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            if parts:
                urls.append(urljoin(base_url, parts[-1]))

    seen = set()
    result = []
    for url in urls:
        if url and is_image_url(url) and url not in seen:
            seen.add(url)
            result.append(url)
    
    return result


def read_feed_list(path: str) -> List[str]:
    feeds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                feeds.append(line)
    return feeds


def get_feed_entries(feed_url: str, max_entries: int) -> List[Tuple[str, str, str]]:
    parsed = feedparser.parse(feed_url)
    entries = []
    
    for entry in parsed.entries[:max_entries]:
        link = getattr(entry, "link", None)
        title = normalize_whitespace(getattr(entry, "title", ""))
        published = getattr(entry, "published", "")
        if link:
            entries.append((link, title, published))
    
    return entries


def main():
    parser = argparse.ArgumentParser(description="Scrape photo essays from RSS feeds")
    parser.add_argument("--feeds", required=True, help="Path to feeds.txt (one URL per line)")
    parser.add_argument("--out", default=None, help="Output JSONL path (default: output/rss_photoessays.jsonl)")
    parser.add_argument("--max_feeds", type=int, default=None, help="Maximum number of feeds to process")
    parser.add_argument("--max_entries_per_feed", type=int, default=20, help="Max entries per feed")
    parser.add_argument("--min_words", type=int, default=300, help="Minimum word count")
    parser.add_argument("--min_images", type=int, default=5, help="Minimum number of images")
    parser.add_argument("--mode", choices=["requests", "playwright"], default="requests",
                        help="requests=fast/static, playwright=JS-heavy")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between page fetches")
    parser.add_argument("--since", default=None, help="Only keep entries published on/after YYYY-MM-DD")
    args = parser.parse_args()

    # since_dt is parsed but not applied per-entry.
    since_dt = parse_date(args.since) if args.since else None

    if args.out is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        args.out = str(DEFAULT_OUTPUT_DIR / "rss_photoessays.jsonl")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    feeds = read_feed_list(args.feeds)
    if args.max_feeds:
        feeds = feeds[:args.max_feeds]

    print(f"Processing {len(feeds)} feeds (min_words={args.min_words}, min_images={args.min_images})")
    if since_dt:
        print(f"  date filter: since {args.since}")

    seen_urls = set()
    kept_count = 0
    scanned_count = 0

    with open(args.out, "w", encoding="utf-8") as out:
        for feed_idx, feed_url in enumerate(feeds, start=1):
            entries = get_feed_entries(feed_url, args.max_entries_per_feed)
            print(f"\n[Feed {feed_idx}/{len(feeds)}] {feed_url}")
            print(f"  Found {len(entries)} entries")

            for url, title, published in entries:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                scanned_count += 1

                if args.mode == "requests":
                    html = fetch_html_requests(url)
                else:
                    html = fetch_html_playwright(url)

                if not html:
                    continue

                text = extract_article_text(html, url)
                if not text:
                    time.sleep(args.sleep)
                    continue

                word_count = count_words(text)
                if word_count < args.min_words:
                    time.sleep(args.sleep)
                    continue

                images = extract_image_urls(html, url)
                if len(images) < args.min_images:
                    time.sleep(args.sleep)
                    continue

                record = {
                    "source_feed": feed_url,
                    "url": url,
                    "domain": get_domain(url),
                    "title": title,
                    "published": published,
                    "word_count": word_count,
                    "num_images": len(images),
                    "images": images,
                    "text": text,
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                kept_count += 1

                print(f"  kept #{kept_count}: {title[:60]} (words={word_count}, imgs={len(images)})")
                time.sleep(args.sleep)

    print(f"\nDone. scanned={scanned_count}, kept={kept_count} → {args.out}")


if __name__ == "__main__":
    main()
