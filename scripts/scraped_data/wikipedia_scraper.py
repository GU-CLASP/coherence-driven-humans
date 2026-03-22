#!/usr/bin/env python3
"""Scrape Wikipedia and Wikibooks pages into JSONL."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

USER_AGENT = "wikipedia-scraper/0.2 (research)"

PROJECTS = {
    "enwiki": {
        "api": "https://en.wikipedia.org/w/api.php",
        "base": "https://en.wikipedia.org/wiki/",
    },
    "enwikibooks": {
        "api": "https://en.wikibooks.org/w/api.php",
        "base": "https://en.wikibooks.org/wiki/",
    },
}

NS_MAIN = 0
NS_TALK = 1

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def make_wiki_url(base: str, title: str) -> str:
    return base + quote(title.replace(" ", "_"))


def api_request(api_url: str, session: requests.Session, params: Dict, 
                retries: int = 5, timeout: int = 20) -> Dict:
    params = {**params, "format": "json", "formatversion": "2"}

    backoff = 1.0
    for attempt in range(retries):
        try:
            response = session.get(api_url, params=params, timeout=timeout,
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
        "span.mw-editsection", "sup.reference", "div.navbox",
        "div.catlinks", "style", "script"
    ]
    for selector in selectors_to_remove:
        for node in root.select(selector):
            node.decompose()

    blocks = []
    for element in root.find_all(["p", "li"]):
        text = element.get_text(" ", strip=True)
        if text:
            blocks.append(text)

    result = []
    prev = None
    for block in blocks:
        if block != prev:
            result.append(block)
        prev = block

    return "\n\n".join(result).strip()


def iter_recent_changes(api_url: str, session: requests.Session,
                        start_iso: str, end_iso: str, namespace: int,
                        rc_type: str, max_items: Optional[int] = None) -> Iterable[Dict]:
    """Yield recent changes for one namespace."""
    continuation = None
    yielded = 0

    while True:
        params = {
            "action": "query",
            "list": "recentchanges",
            "rcstart": start_iso,
            "rcend": end_iso,
            "rcdir": "newer",
            "rcnamespace": namespace,
            "rctype": rc_type,
            "rcprop": "title|timestamp|ids|sizes|flags",
            "rclimit": 500,
        }
        if continuation:
            params.update(continuation)

        data = api_request(api_url, session, params)
        changes = data.get("query", {}).get("recentchanges", [])
        
        for change in changes:
            if change.get("title") and change.get("timestamp"):
                yield change
                yielded += 1
                if max_items and yielded >= max_items:
                    return

        continuation = data.get("continue")
        if not continuation:
            break
        time.sleep(0.05)


def parse_page(api_url: str, session: requests.Session, title: str,
               section: Optional[int] = None) -> Tuple[int, str]:
    """Return page id and parsed text."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "redirects": 1,
        "disabletoc": 1,
    }
    if section is not None:
        params["section"] = section

    data = api_request(api_url, session, params)
    parse_data = data.get("parse", {})
    return parse_data.get("pageid", -1), html_to_text(parse_data.get("text", ""))


def get_page_creation_date(api_url: str, session: requests.Session, title: str) -> Optional[str]:
    """Return the first revision timestamp for a page."""
    data = api_request(api_url, session, {
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


def get_page_images(api_url: str, session: requests.Session, title: str) -> List[str]:
    data = api_request(api_url, session, {
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


def resolve_image_info(api_url: str, session: requests.Session, file_titles: List[str]) -> Dict[str, Dict]:
    info = {}
    
    for i in range(0, len(file_titles), 25):
        batch = file_titles[i:i+25]
        data = api_request(api_url, session, {
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


def main():
    parser = argparse.ArgumentParser(description="Extract text from Wikipedia/Wikibooks")
    parser.add_argument("--project", choices=PROJECTS.keys(), required=True,
                        help="Wiki project (enwiki or enwikibooks)")
    parser.add_argument("--kind", choices=["wikipedia_lead", "wikibooks_page", "wikipedia_talk"],
                        required=True, help="Type of content to extract")
    parser.add_argument("--start", required=True, 
                        help="Start timestamp (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--end", required=True,
                        help="End timestamp (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--out", default=None,
                        help="Output JSONL path (default: output/<project>_<kind>.jsonl)")
    parser.add_argument("--rctype", default="new|edit",
                        help="Recent changes type: new, edit, or new|edit")
    parser.add_argument("--max_pages", type=int, default=200,
                        help="Maximum pages to process")
    parser.add_argument("--sleep", type=float, default=0.15,
                        help="Delay between requests")
    parser.add_argument("--progress_every", type=int, default=10,
                        help="Progress update frequency")
    parser.add_argument("--with_images", action="store_true",
                        help="Include image URLs in output")
    parser.add_argument("--created_after", default=None,
                        help="Only include pages created on/after this date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)")
    args = parser.parse_args()

    project = PROJECTS[args.project]
    api_url = project["api"]
    base_url = project["base"]

    created_after_ts = None
    if args.created_after:
        if "T" not in args.created_after:
            created_after_ts = args.created_after + "T00:00:00Z"
        else:
            created_after_ts = args.created_after
        print(f"Filtering: only pages created on/after {created_after_ts}")

    if args.kind == "wikipedia_lead":
        namespace = NS_MAIN
        section = 0
    elif args.kind == "wikibooks_page":
        namespace = NS_MAIN
        section = None
    elif args.kind == "wikipedia_talk":
        namespace = NS_TALK
        section = None
    else:
        raise ValueError(f"Unknown kind: {args.kind}")

    if args.out is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        args.out = str(DEFAULT_OUTPUT_DIR / f"{args.project}_{args.kind}.jsonl")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"{args.project} / {args.kind}: {args.start} to {args.end} → {args.out}")

    with requests.Session() as session, open(args.out, "w", encoding="utf-8") as f:
        print("Collecting pages from recent changes...")
        seen = set()
        candidates: List[Dict] = []
        
        for change in iter_recent_changes(api_url, session, args.start, args.end,
                                          namespace, args.rctype):
            title = change["title"]
            if title in seen:
                continue
            seen.add(title)
            candidates.append(change)
            if len(candidates) >= args.max_pages:
                break

        print(f"Found {len(candidates)} candidate pages")

        start_time = time.time()
        written = 0
        skipped_old = 0

        for idx, change in enumerate(candidates, start=1):
            title = change["title"]
            timestamp = change["timestamp"]
            rev_id = change.get("revid")

            if idx == 1 or idx % args.progress_every == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (len(candidates) - idx) / rate if rate > 0 else float("inf")
                print(f"[{idx}/{len(candidates)}] written={written} skipped_old={skipped_old} rate={rate:.2f}/s "
                      f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)}")

            if created_after_ts:
                try:
                    creation_date = get_page_creation_date(api_url, session, title)
                    if creation_date and creation_date < created_after_ts:
                        skipped_old += 1
                        time.sleep(args.sleep)
                        continue
                except Exception as e:
                    print(f"[warn] Failed to get creation date for '{title}': {e}")

            try:
                pageid, text = parse_page(api_url, session, title, section=section)
            except Exception as e:
                print(f"[warn] Failed to parse '{title}': {e}")
                time.sleep(args.sleep)
                continue

            record = {
                "source": args.project,
                "kind": args.kind,
                "title": title,
                "pageid": pageid,
                "rev_id": rev_id,
                "timestamp": timestamp,
                "url": make_wiki_url(base_url, title),
                "text": text,
            }

            if args.with_images:
                try:
                    file_titles = get_page_images(api_url, session, title)
                    excluded = ["Commons-logo", "Wiki-logo", "Wikimedia", "Icon", "Symbol",
                                "Ambox", "Edit-clear", "Question_book", "Padlock"]
                    file_titles = [ft for ft in file_titles if not any(x in ft for x in excluded)]
                    image_info = resolve_image_info(api_url, session, file_titles)
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
