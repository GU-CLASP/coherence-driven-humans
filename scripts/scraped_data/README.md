# Technical Details: scraped multimodal data

## General Description

We scrape paired texts and image sequences from teh web.
This data is used for calculation of perplexity of open-source models on unseen general-domain multimodal data.
We use three open-source VLMs as evaluators (Qwen3-VL, Llama 4 Scout, and InternVL-3) on both VWP storytelling data and this additional set of unseen multimodal web documents.
In the scraped setting, evaluator models received images and text only, with no instruction or system prompt.
Perplexity was computed from log-probabilities over text tokens, excluding control and vision tokens.

## Data Sources

We collected data from three Wikimedia-related sources:

1. **Wikipedia article leads** — first paragraphs of Wikipedia articles (project `enwiki`, kind `wikipedia_lead`)
2. **Wikinews articles** — full news articles from en.wikinews.org, retrieved by publication date category
3. **RSS photo-essay pages** — articles from RSS feeds (listed in `feeds.txt`) with text and multiple images

## Filtering Criteria

To align the collected data with VLM knowledge cut-offs and to keep the structure closer to VWP, the following automatic constraints were applied:

- **(a) Publication date**: not earlier than **2025-11-26** (aligns with model knowledge cut-off boundaries)
- **(b) RSS items**: at least **300 words** and at least **9 images** per article
- **(c) Non-content image filtering**: logos, navigation icons, UI elements, and decorative images were removed automatically (see `process_scraped_data.py`)
- **(d) After all filtering**: only items with at least 1 remaining content image were kept

After filtering, **46 items** (image sequence(s) with associated text) remained.

To reproduce, run all commands from the `scripts/scraped_data/` directory.

### Install dependencies

```bash
pip install requests beautifulsoup4 lxml feedparser trafilatura
```

### Scrape Wikipedia article leads

```bash
python wikipedia_scraper.py \
    --project enwiki \
    --kind wikipedia_lead \
    --start 2025-11-26T00:00:00Z \
    --end 2026-02-19T00:00:00Z \
    --created_after 2025-11-25 \
    --with_images
```

The `--created_after 2025-11-25` flag filters out old pages that were simply re-edited in the window (the Recent Changes API returns pages *edited*, not necessarily *created*, in the date range).

### Scrape Wikinews articles

```bash
python wikinews_scraper.py \
    --start 2025-11-26 \
    --end 2026-02-19 \
    --created_after 2025-11-25 \
    --with_images
```

### Scrape RSS photo-essay pages

```bash
python rss_scraper.py \
    --feeds feeds.txt \
    --min_words 300 \
    --min_images 9 \
    --since 2025-11-25
```

### Process and filter all scraped data

```bash
python process_scraped_data.py \
    --min_images 1 \
    --min_text_length 100
```

This produces `output/processed_data.jsonl` with 46 entries and assigns each a unique integer `id`.
Optionally add `--download_images` to download image files locally.