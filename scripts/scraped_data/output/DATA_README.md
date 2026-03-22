## Data Format

Each entry in `processed_data.jsonl` contains:

```json
{
  "id": 1,
  "text": "Article text content...",
  "images": [
    {
      "file_title": "File:Example.jpg",
      "url": "https://...",
      "mime": "image/jpeg",
      "width": 1920,
      "height": 1080
    }
  ],
  "source_title": "Original Article Title",
  "source_url": "https://...",
  "original_image_count": 8,
  "filtered_image_count": 5
}
```

## Image Filtering

Non-content images were automatically filtered out, including:
- Logos and icons (Wikipedia, Commons, etc.)
- UI elements (arrows, buttons, edit icons)
- Very small images (< 50x50 pixels)
- Template/infobox decorations

The `original_image_count` vs `filtered_image_count` fields show how many images were removed.

## Usage

```python
import json

with open("processed_data.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        text = entry["text"]
        images = entry["images"]
        # Compute perplexity, etc.
```