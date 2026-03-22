#!/usr/bin/env python3
"""Scrape Wikinews articles by publication date into JSONL."""

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

API_URL = "https://en.wikinews.org/w/api.php"
BASE_URL = "https://en.wikinews.org/wiki/"
USER_AGENT = "wikinews-scraper/0.2 (research)"

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def get_wiki_url(title: str) -> str:
    return BASE_URL + quote(title.replace(" ", "_"))


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def get_category_name(dt: datetime) -> str:
    return f"Category:{dt.strftime('%B')} {dt.day}, {dt.year}"


def api_request(session: requests.Session, params: Dict, retries: int = 5, timeout: int = 20) -> Dict:
    params = {**params, "format": "json", "formatversion": "2"}
    
    backoff = 1.0
    for attempt in range(retries):
        try:
            response = session.get(API_URL, params=params, timeout=timeout, 
                                   headers={"User-Agent": USER_AGENT})
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"API request failed after {retries} attempts: {e}")
            time.sleep(backoff)
            backoff *= 2
    return {}


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one("div.mw-parser-output") or soup

    selectors_to_remove = [
        "table", "ol.references", "div.reflist", "div.thumb", 
        "div.metadata", "span.mw-editsection", "sup.reference", 
        "div.navbox", "div.catlinks"
    ]
    for selector in selectors_to_remove:
        for node in root.select(selector):
            node.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in root.find_all("p")]
    return "\n\n".join(p for p in paragraphs if p).strip()


def get_articles_for_date(session: requests.Session, category: str) -> List[str]:
    titles = []
    continuation = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmnamespace": 0,
            "cmlimit": 500,
            "cmprop": "title",
        }
        if continuation:
            params.update(continuation)

        data = api_request(session, params)
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members if m.get("title"))

        continuation = data.get("continue")
        if not continuation:
            break
        time.sleep(0.05)

    seen = set()
    return [t for t in titles if not (t in seen or seen.add(t))]


def fetch_article_text(session: requests.Session, title: str) -> Tuple[int, str]:
    data = api_request(session, {
        "action": "parse",
        "page": title,
        "prop": "text",
        "redirects": 1,
        "disabletoc": 1,
    })
    parse_data = data.get("parse", {})
    return parse_data.get("pageid", -1), html_to_text(parse_data.get("text", ""))


def get_article_images(session: requests.Session, title: str) -> List[str]:
    data = api_request(session, {
        "action": "query",
        "prop": "images",
        "titles": title,
        "imlimit": "max",
        "redirects": 1,
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return []
    
    images = pages[0].get("images", [])
    return sorted({img["title"] for img in images if img.get("title", "").startswith("File:")})


def resolve_image_info(session: requests.Session, file_titles: List[str]) -> Dict[str, Dict]:
    info = {}
    
    for i in range(0, len(file_titles), 25):
        batch = file_titles[i:i+25]
        data = api_request(session, {
            "action": "query",
            "prop": "imageinfo",
            "titles": "|".join(batch),
            "iiprop": "url|mime|size|dimensions",
            "redirects": 1,
        })
        
        for page in data.get("query", {}).get("pages", []):
            title = page.get("title")
            image_info = page.get("imageinfo", [])
            if title and image_info and image_info[0].get("url"):
                ii = image_info[0]
                info[title] = {
                    "file_title": title,
                    "url": ii.get("url"),
                    "mime": ii.get("mime"),
                    "width": ii.get("width"),
                    "height": ii.get("height"),
                    "size_bytes": ii.get("size"),
                }
        time.sleep(0.05)
    
    return info


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins:02d}m"


def get_page_creation_date(session: requests.Session, title: str) -> Optional[str]:
    """Return the first revision timestamp for a page."""
    data = api_request(session, {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvprop": "timestamp",
        "rvlimit": 1,
        "rvdir": "newer",
        "redirects": 1,
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    
    revisions = pages[0].get("revisions", [])
    if not revisions:
        return None
    
    return revisions[0].get("timestamp")


def main():
    parser = argparse.ArgumentParser(description="Scrape Wikinews articles by publication date")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument("--out", default=None, help="Output JSONL path (default: output/wikinews_<start>_<end>.jsonl)")
    parser.add_argument("--with_images", action="store_true", help="Include image URLs")
    parser.add_argument("--sleep", type=float, default=0.15, help="Delay between requests (seconds)")
    parser.add_argument("--max_articles", type=int, default=None, help="Maximum articles to fetch")
    parser.add_argument("--progress_every", type=int, default=10, help="Progress update frequency")
    parser.add_argument("--created_after", default=None,
                        help="Only include pages created on/after this date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)")
    args = parser.parse_args()

    created_after_ts = None
    if args.created_after:
        if "T" not in args.created_after:
            created_after_ts = args.created_after + "T00:00:00Z"
        else:
            created_after_ts = args.created_after
        print(f"Filtering: only pages created on/after {created_after_ts}")

    start_dt = parse_date(args.start)
    end_dt = parse_date(args.end)

    if args.out is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        args.out = str(DEFAULT_OUTPUT_DIR / f"wikinews_{args.start}_{args.end}.jsonl")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with requests.Session() as session, open(args.out, "w", encoding="utf-8") as f:
        print(f"Collecting articles from {args.start} to {args.end}...")
        all_articles = []  # (title, published_date)
        
        current_day = start_dt
        while current_day < end_dt:
            category = get_category_name(current_day)
            pub_date = format_date(current_day)
            titles = get_articles_for_date(session, category)
            
            print(f"  {pub_date}: {len(titles)} articles")
            all_articles.extend((title, pub_date) for title in titles)
            
            current_day += timedelta(days=1)
            time.sleep(0.05)

        first_seen = {}
        for title, date in all_articles:
            if title not in first_seen:
                first_seen[title] = date

        articles = sorted(first_seen.items(), key=lambda x: x[1])
        print(f"\nTotal unique articles: {len(articles)}")

        start_time = time.time()
        written = 0
        skipped_old = 0

        for idx, (title, pub_date) in enumerate(articles, start=1):
            if args.max_articles and written >= args.max_articles:
                break

            if idx == 1 or idx % args.progress_every == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (len(articles) - idx) / rate if rate > 0 else float("inf")
                print(f"[{idx}/{len(articles)}] written={written} skipped_old={skipped_old} rate={rate:.2f}/s "
                      f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)}")

            if created_after_ts:
                try:
                    creation_date = get_page_creation_date(session, title)
                    if creation_date and creation_date < created_after_ts:
                        skipped_old += 1
                        time.sleep(args.sleep)
                        continue
                except Exception as e:
                    print(f"[warn] Failed to get creation date for '{title}': {e}")

            try:
                pageid, text = fetch_article_text(session, title)
            except Exception as e:
                print(f"[warn] Failed to parse '{title}': {e}")
                time.sleep(args.sleep)
                continue

            record = {
                "title": title,
                "pageid": pageid,
                "published_date": pub_date,
                "url": get_wiki_url(title),
                "text": text,
            }

            if args.with_images:
                try:
                    file_titles = get_article_images(session, title)
                    excluded = ["Wikinews", "Commons-logo", "Powered_by"]
                    file_titles = [ft for ft in file_titles if not any(x in ft for x in excluded)]
                    image_info = resolve_image_info(session, file_titles)
                    record["images"] = [image_info[ft] for ft in file_titles if ft in image_info]
                except Exception as e:
                    print(f"[warn] Failed to get images for '{title}': {e}")
                    record["images"] = []

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            time.sleep(args.sleep)

        print(f"\nDone! Wrote {written} records to {args.out}")


if __name__ == "__main__":
    main()
