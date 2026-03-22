#!/usr/bin/env python3
"""
Process Scraped Data

Processes JSONL files from the scrapers, filters out UI/template images,
assigns unique IDs, and optionally downloads images.

Output:
- A clean JSONL file with id, text, and filtered images
- (Optional) Downloaded images named as {id}_{img_num}.{ext}

Example usage:
    # Process all JSONL files in output/
    python process_scraped_data.py

    # Process and download images
    python process_scraped_data.py --download_images

    # Process specific files
    python process_scraped_data.py --input output/wikinews*.jsonl

    # Custom output paths
    python process_scraped_data.py --out processed_data.jsonl --images_dir images/

Requirements: requests
"""

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

# Default paths
DEFAULT_INPUT_DIR = Path(__file__).parent / "output"
DEFAULT_OUTPUT_FILE = Path(__file__).parent / "output" / "processed_data.jsonl"
DEFAULT_IMAGES_DIR = Path(__file__).parent / "output" / "images"

USER_AGENT = "image-downloader/0.1 (research)"

# ============================================================================
# Image filtering patterns
# ============================================================================
# These patterns identify UI elements, icons, logos, and template images
# that are not actual content images from the articles.

EXCLUDE_PATTERNS = [
    # Wikimedia project logos
    r"logo",
    r"Logo",
    r"Commons-logo",
    r"Wiki-logo",
    r"Wikimedia",
    r"Wikipedia-logo",
    r"Wiktionary-logo",
    r"Wikinews-logo",
    r"Wikibooks-logo",
    r"Wikiquote-logo",
    r"Wikisource-logo",
    r"Wikispecies-logo",
    r"Wikiversity-logo",
    r"Wikivoyage-logo",
    r"Wikidata-logo",
    r"MediaWiki-logo",
    
    # Icons and UI elements
    r"Icon",
    r"icon\.",
    r"_icon",
    r"Symbol",
    r"Pictogram",
    r"Button",
    r"Arrow",
    r"Bullet",
    
    # Message boxes and notices
    r"Ambox",          # Article message box
    r"Cmbox",          # Category message box
    r"Imbox",          # Image message box
    r"Tmbox",          # Talk page message box
    r"Ombox",          # Other pages message box
    r"Mbox",
    r"Question_book",
    r"Edit-clear",
    r"Information_icon",
    
    # Protection and status icons
    r"Padlock",
    r"Lock-",
    r"Semi-protection",
    r"Full-protection",
    
    # Navigation and interface
    r"External[_-]link",
    r"Searchtool",
    r"Magnifying[_-]glass",
    r"Portal-puzzle",
    r"Folder[_-]",
    r"Disambig",
    r"Stub[_-]",
    
    # Icon sets (common in Wikipedia)
    r"Nuvola",
    r"Crystal[_-]",
    r"Gnome-",
    r"Tango-",
    r"Oxygen-",
    r"OOjs_UI",
    r"Ooui-",
    
    # Map markers and location dots
    r"Red_pog",
    r"Green_pog",
    r"Blue_pog",
    r"Location_dot",
    r"Map[_-]marker",
    r"Geolocator",
    
    # Status/trend indicators
    r"Increase",
    r"Decrease",
    r"Steady",
    r"Green[-_]?check",
    r"X[-_]?mark",
    r"Yes[-_]?check",
    r"Checkmark",
    
    # Emoji and emoticons
    r"Emojione",
    r"Twemoji",
    r"Noto_Emoji",
    r"Emoji_",
    
    # Flags (often UI, but could be content - be careful)
    # r"Flag_of",  # Commented out - flags might be relevant content
    
    # Sound/media icons
    r"Speaker_Icon",
    r"Gnome-mime-sound",
    r"Loudspeaker",
    
    # Rating stars
    r"Star_full",
    r"Star_empty",
    r"Star_half",
    
    # Generic template/portal indicators
    r"[_\s]template[_\s]",
    r"[_\s]sidebar[_\s]",
    r"[_\s]infobox[_\s]",
    r"^Portal[_-]",             # Portal icons
    r"Wikinews[_\s]",           # Wikinews template images
    r"[_\s]stub\.",             # Stub icons
    
    # Other common non-content patterns
    r"Wikiproject",
    r"Wikipe-tan",
    r"Powered_by",
    r"Copyright",
    r"PD-icon",
    r"CC-",            # Creative Commons badges
    r"Cc-by",
    r"GFDL",
    r"GPL",
    
    # File type indicators often used as icons
    r"PDF_file",
    r"Gnome-mime-application",
    
    # Icon fonts and libraries - any file with "font" in the name is likely an icon
    r"[Ff]ont[_\s]",   # Catches "Font Awesome", "Font_", "font_" etc.
    r"Material[_\s]?Icons?",
    r"Ionicons?",
    r"Feather[_\s]?icons?",
    r"Bootstrap[_\s]?Icons?",
    r"Heroicons?",
    r"Octicons?",
    r"fa-solid",       # Font Awesome class-based names
    r"fa-regular",
    r"fa-brands",
    r"\d+[_\s]solid[_\s]",   # Font Awesome patterns like "5 solid " or "5_solid_"
    r"\d+[_\s]regular[_\s]",
    r"\d+[_\s]brands[_\s]",
]

# Compile patterns for efficiency
EXCLUDE_REGEX = re.compile("|".join(EXCLUDE_PATTERNS), re.IGNORECASE)


def get_filename_from_url(url: str) -> str:
    """Extract just the filename from a URL (without domain/path)."""
    if not url:
        return ""
    parsed = urlparse(url)
    # Get the last part of the path
    path = parsed.path
    filename = path.split("/")[-1] if "/" in path else path
    return filename


def get_filename_from_title(file_title: str) -> str:
    """Extract filename from wiki file title like 'File:Example.jpg'."""
    if not file_title:
        return ""
    # Remove "File:" prefix if present
    if file_title.startswith("File:"):
        return file_title[5:]
    return file_title


def normalize_image(img) -> Dict:
    """
    Normalize image to a dictionary format.
    Handles both plain URL strings (from RSS scraper) and dicts (from Wiki scrapers).
    """
    if isinstance(img, str):
        # Plain URL string from RSS scraper
        return {"url": img, "file_title": ""}
    elif isinstance(img, dict):
        return img
    else:
        return {"url": str(img), "file_title": ""}


def should_keep_image(image_info: Dict) -> bool:
    """
    Determine if an image should be kept based on its file title/URL.
    Returns True if the image appears to be actual content (not UI/icon).
    
    Only checks the FILENAME part, not the full URL domain (to avoid
    filtering out valid images just because they're hosted on wikimedia.org).
    """
    # Normalize first if needed
    if isinstance(image_info, str):
        image_info = normalize_image(image_info)
    
    file_title = image_info.get("file_title", "")
    url = image_info.get("url", "")
    
    # Extract just the filename parts - NOT the full URL with domain
    filename_from_title = get_filename_from_title(file_title)
    filename_from_url = get_filename_from_url(url)
    
    # Check only the filenames, not full URLs
    text_to_check = f"{filename_from_title} {filename_from_url}"
    
    if EXCLUDE_REGEX.search(text_to_check):
        return False
    
    # Additional heuristics
    
    # Very small images are likely icons (if we have size info)
    width = image_info.get("width", 0)
    height = image_info.get("height", 0)
    if width and height and width < 50 and height < 50:
        return False
    
    # SVGs that are small are almost certainly icons; larger ones may be diagrams or maps
    mime = image_info.get("mime", "")
    if mime == "image/svg+xml":
        # Only keep SVGs if they're larger (likely diagrams/maps)
        if width and height and (width < 200 or height < 200):
            return False
    
    return True


def get_extension_from_url(url: str, mime: Optional[str] = None) -> str:
    """Extract file extension from URL or mime type."""
    # Try from URL first
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    if path.endswith(".jpg") or path.endswith(".jpeg"):
        return "jpg"
    elif path.endswith(".png"):
        return "png"
    elif path.endswith(".gif"):
        return "gif"
    elif path.endswith(".webp"):
        return "webp"
    elif path.endswith(".svg"):
        return "svg"
    
    # Fall back to mime type
    if mime:
        mime_map = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp",
            "image/svg+xml": "svg",
        }
        return mime_map.get(mime, "jpg")
    
    return "jpg"  # Default


def download_image(url: str, dest_path: Path, timeout: int = 30) -> bool:
    """Download an image to the specified path. Returns True on success."""
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"[warn] Failed to download {url}: {e}")
        return False


# Patterns indicating the text is not real article content
INVALID_TEXT_PATTERNS = [
    # Retracted articles
    r"retract(ed|ion)",
    r"this article has been retracted",
    r"failed to comply",
    r"copyright policy",
    r"article text can be found here",
    
    # Deleted/removed content
    r"this (article|page) (has been|was) (deleted|removed)",
    r"content (has been|was) removed",
    
    # Stub indicators
    r"this article is a stub",
    r"you can help .* by expanding",
]

INVALID_TEXT_REGEX = re.compile("|".join(INVALID_TEXT_PATTERNS), re.IGNORECASE)

# Date-only patterns (text that's just a date with no content)
DATE_ONLY_PATTERNS = [
    # "Thursday, December 25, 2025" or similar
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"\d{1,2},?\s+\d{4}\.?$",
    
    # ISO date
    r"^\d{4}-\d{2}-\d{2}$",
    
    # Other date formats
    r"^\d{1,2}/\d{1,2}/\d{4}$",
    r"^\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$",
]

DATE_ONLY_REGEX = re.compile("|".join(DATE_ONLY_PATTERNS), re.IGNORECASE)

# Text artifact patterns to clean
TEXT_ARTIFACT_PATTERNS = [
    # Cite errors
    r"Cite error:.*?(?=\.|$)",
    r"\(see the help page\s*\)\.?",
    # Citation needed
    r"\[\s*citation needed\s*\]",
    r"\[\s*clarification needed\s*\]",
    r"\[\s*when\?\s*\]",
    r"\[\s*who\?\s*\]",
    r"\[\s*where\?\s*\]",
    r"\[\s*which\?\s*\]",
    # Reflist/references
    r"<ref[^>]*>.*?</ref>",
    r"\{\{reflist[^}]*\}\}",
    # Translate request (from RSS)
    r"Can you help us translate this article\?.*?Start translation",
]

TEXT_ARTIFACT_REGEX = re.compile("|".join(TEXT_ARTIFACT_PATTERNS), re.IGNORECASE | re.DOTALL)


def clean_text_artifacts(text: str) -> str:
    """Remove Wikipedia/RSS artifacts from text."""
    text = TEXT_ARTIFACT_REGEX.sub("", text)
    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # Fix spacing before punctuation
    return text.strip()


def is_valid_text(text: str) -> Tuple[bool, str]:
    """
    Check if text is valid article content.
    Returns (is_valid, reason) tuple.
    """
    if not text:
        return False, "empty"
    
    text = text.strip()
    
    # Check if text is too short (probably just metadata)
    if len(text) < 50:
        return False, "too_short"
    
    # Check if it's just a date
    if DATE_ONLY_REGEX.match(text):
        return False, "date_only"
    
    # Check for retracted/invalid content markers
    if INVALID_TEXT_REGEX.search(text):
        return False, "retracted_or_invalid"
    
    # Count actual words (not just whitespace/punctuation)
    word_count = len(re.findall(r'\b[a-zA-Z]{2,}\b', text))
    if word_count < 20:
        return False, "insufficient_words"
    
    return True, "valid"


def process_entry(entry: Dict, entry_id: int) -> Dict:
    """
    Process a single entry: extract text, filter images, assign ID.
    Returns the processed entry.
    """
    # Get text (different scrapers use slightly different formats)
    text = entry.get("text", "")
    
    # Get images, normalize to dict format, and filter
    raw_images = entry.get("images", [])
    normalized_images = [normalize_image(img) for img in raw_images]
    filtered_images = [img for img in normalized_images if should_keep_image(img)]
    
    # Build processed entry
    processed = {
        "id": entry_id,
        "text": text,
        "images": filtered_images,
        # Keep some metadata for reference
        "source_title": entry.get("title", ""),
        "source_url": entry.get("url", ""),
        "original_image_count": len(raw_images),
        "filtered_image_count": len(filtered_images),
    }
    
    return processed


def main():
    parser = argparse.ArgumentParser(description="Process scraped data and filter images")
    parser.add_argument("--input", nargs="*", default=None,
                        help="Input JSONL files (glob patterns supported). Default: output/*.jsonl")
    parser.add_argument("--out", default=None,
                        help=f"Output JSONL path (default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("--images_dir", default=None,
                        help=f"Directory for downloaded images (default: {DEFAULT_IMAGES_DIR})")
    parser.add_argument("--download_images", action="store_true",
                        help="Download filtered images")
    parser.add_argument("--sleep", type=float, default=0.1,
                        help="Delay between image downloads (seconds)")
    parser.add_argument("--min_images", type=int, default=1,
                        help="Minimum images required to include an entry (default: 1)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum images allowed per entry (default: no limit)")
    parser.add_argument("--min_text_length", type=int, default=100,
                        help="Minimum text length to include an entry (default: 100 chars)")
    parser.add_argument("--clean_text", action="store_true",
                        help="Remove Wikipedia artifacts from text (cite errors, etc.)")
    args = parser.parse_args()

    # Set default paths
    if args.out is None:
        args.out = str(DEFAULT_OUTPUT_FILE)
    if args.images_dir is None:
        args.images_dir = str(DEFAULT_IMAGES_DIR)

    # Find input files
    if args.input is None:
        input_pattern = str(DEFAULT_INPUT_DIR / "*.jsonl")
        input_files = glob.glob(input_pattern)
    else:
        input_files = []
        for pattern in args.input:
            input_files.extend(glob.glob(pattern))
    
    # Exclude the output file from inputs
    input_files = [f for f in input_files if os.path.basename(f) != os.path.basename(args.out)]
    
    if not input_files:
        print("No input files found!")
        return

    print(f"Input files ({len(input_files)}):")
    for f in input_files:
        print(f"  {f}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.download_images:
        os.makedirs(args.images_dir, exist_ok=True)

    # Process all files
    entry_id = 0
    total_entries = 0
    kept_entries = 0
    total_images = 0
    filtered_images = 0
    downloaded_images = 0
    skipped_reasons = {}  # Track why entries were skipped

    with open(args.out, "w", encoding="utf-8") as out_f:
        for input_file in input_files:
            print(f"Processing {input_file}...")
            
            with open(input_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    total_entries += 1
                    
                    # Skip entries without images
                    if "images" not in entry or not entry["images"]:
                        skipped_reasons["no_images"] = skipped_reasons.get("no_images", 0) + 1
                        continue
                    
                    # Validate text content BEFORE processing
                    text = entry.get("text", "")
                    text_valid, reason = is_valid_text(text)
                    if not text_valid:
                        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                        continue
                    
                    entry_id += 1
                    processed = process_entry(entry, entry_id)
                    
                    total_images += processed["original_image_count"]
                    filtered_images += processed["filtered_image_count"]
                    
                    # Apply additional filters
                    if len(processed["text"]) < args.min_text_length:
                        skipped_reasons["text_too_short"] = skipped_reasons.get("text_too_short", 0) + 1
                        continue
                    if processed["filtered_image_count"] < args.min_images:
                        skipped_reasons["insufficient_images"] = skipped_reasons.get("insufficient_images", 0) + 1
                        continue
                    if args.max_images and processed["filtered_image_count"] > args.max_images:
                        skipped_reasons["too_many_images"] = skipped_reasons.get("too_many_images", 0) + 1
                        continue
                    
                    # clean_text runs regex substitutions to remove citation artifacts;
                    # skip by default since it's slow for large datasets
                    if args.clean_text:
                        processed["text"] = clean_text_artifacts(processed["text"])
                    
                    kept_entries += 1
                    
                    # Download images if requested
                    if args.download_images and processed["images"]:
                        for img_idx, img in enumerate(processed["images"], start=1):
                            url = img.get("url")
                            if not url:
                                continue
                            
                            ext = get_extension_from_url(url, img.get("mime"))
                            filename = f"{entry_id}_{img_idx}.{ext}"
                            dest_path = Path(args.images_dir) / filename
                            
                            if download_image(url, dest_path):
                                downloaded_images += 1
                                # Add local path to image info
                                img["local_path"] = str(dest_path)
                            
                            time.sleep(args.sleep)
                    
                    # Write processed entry
                    out_f.write(json.dumps(processed, ensure_ascii=False) + "\n")
                    out_f.flush()
                    
                    if kept_entries % 50 == 0:
                        print(f"  {total_entries} scanned, {kept_entries} kept...")

    print(f"\nscanned={total_entries} valid={entry_id} kept={kept_entries} "
          f"images_in={total_images} images_out={filtered_images}" +
          (f" downloaded={downloaded_images}" if args.download_images else ""))
    if skipped_reasons:
        for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
            print(f"  skipped ({reason}): {count}")
    print(f"→ {args.out}")
    if args.download_images:
        print(f"→ {args.images_dir}/")


if __name__ == "__main__":
    main()
