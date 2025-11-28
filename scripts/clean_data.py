#!/usr/bin/env python3
"""Clean model outputs using model-specific cleaning functions.

Example usage:
    python clean_data.py \
        --input-json ../data/post-processing/collected_outputs.json \
        --output-json ../data/post-processing/cleaned_outputs.json
"""

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-json", required=True, 
                       help="Input JSON from collect_data.py (e.g., ../data/post-processing/collected_outputs.json)")
    parser.add_argument("--output-json", required=True, 
                       help="Output cleaned JSON (e.g., ../data/post-processing/cleaned_outputs.json)")
    return parser.parse_args()


def normalize_text_formatting(text):
    """Common text cleanup: remove newlines, fix spacing around [SEP]."""
    cleaned = re.sub(r'\n+', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s*\[SEP\]\s*', ' [SEP] ', cleaned)
    return cleaned.strip()


def clean_qwen3vl_output(text, story_id=None, prompt_type=None, seed=None):
    """QWen3VL: Keep text after </think>."""
    if '</think>' in text:
        cleaned = text.split('</think>')[-1].strip()
    else:
        print(f"⚠️  No </think> separator: Story {story_id}, {prompt_type}, seed {seed}")
        cleaned = text.strip()
    
    if cleaned.startswith("Okay, let's see. I need to write a story"):
        print(f"⚠️  Uncleaned reasoning text found: Story {story_id}, {prompt_type}, seed {seed}")
    
    return normalize_text_formatting(cleaned)


def clean_internvl3_output(text):
    """InternVL3: Clean newlines, add [SEP] if missing."""
    if '[SEP]' in text:
        parts = text.split('[SEP]')
        cleaned_parts = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part and i == len(parts) - 1 and '\n' in part:
                part = part.split('\n')[0].strip()
            if part:
                cleaned_parts.append(part)
        cleaned = ' [SEP] '.join(cleaned_parts)
    else:
        cleaned = text.strip()
        if '\n' in cleaned:
            cleaned = cleaned.split('\n')[0].strip()
        sentences = [s.strip() for s in cleaned.split('.') if s.strip()]
        if sentences:
            formatted = [s + '.' if not s.endswith('.') else s for s in sentences]
            cleaned = ' [SEP] '.join(formatted)
    return normalize_text_formatting(cleaned)


def clean_llama4scout_output(text):
    """Llama4Scout: Clean last part after final [SEP]."""
    parts = text.split('[SEP]')
    cleaned_parts = []
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            if i == len(parts) - 1:
                part = part.split('\n')[0].strip()
            cleaned_parts.append(part)
    result = ' [SEP] '.join(cleaned_parts) if cleaned_parts else text.strip()
    return normalize_text_formatting(result)


def clean_gpt4o_output(text):
    """GPT-4O: Keep as is, just normalize."""
    return normalize_text_formatting(text.strip())


def clean_claude45_output(text):
    """Claude: Remove headers and footers."""
    if text.strip().startswith('[SEP]'):
        lines = text.split('\n')
        story_lines = []
        for line in lines:
            line = line.strip()
            if line and not any(x in line for x in ['I have described', 'The sequence', 'This story', '**']):
                story_lines.append(line)
            elif any(x in line for x in ['I have described', '**']):
                break
        text = ' '.join(story_lines)
    else:
        if '\n' in text:
            text = text[text.find('\n') + 1:]
        if '**' in text:
            text = text[:text.find('**')]
    
    text = re.sub(r'[\s\-\*=_~`#@$%^&()[\]{}|\\:;"\',./<>?/!]+$', '', text.strip())
    return normalize_text_formatting(text)


def clean_model_output(entry):
    """Apply cleaning based on model type."""
    model_type = entry['model_type']
    text = entry['model_output']
    
    # Human outputs don't need cleaning, just normalization
    if model_type == 'human':
        return normalize_text_formatting(text)
    elif model_type == 'qwen3vl':
        return clean_qwen3vl_output(text, entry.get('story_id'), entry.get('prompt_type'), entry.get('seed'))
    elif model_type == 'internvl3':
        return clean_internvl3_output(text)
    elif model_type == 'llama4scout':
        return clean_llama4scout_output(text)
    elif model_type == 'gpt4o':
        return clean_gpt4o_output(text)
    elif model_type == 'claude45':
        return clean_claude45_output(text)
    else:
        print(f"⚠️  Unknown model type: {model_type}")
        return normalize_text_formatting(text)


def main():
    args = parse_args()
    
    print(f"Loading data from {args.input_json}...")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} entries")
    
    # Count entries by model type and prompt type
    model_counts = {}
    for entry in data:
        key = f"{entry['model_type']}_{entry['prompt_type']}"
        model_counts[key] = model_counts.get(key, 0) + 1
    
    print("\nDataset composition:")
    for key, count in sorted(model_counts.items()):
        print(f"  {key}: {count} entries")
    
    print("\nApplying model-specific cleaning functions...")
    cleaned_count = 0
    for entry in data:
        original = entry['model_output']
        cleaned = clean_model_output(entry)
        entry['cleaned_model_output'] = cleaned
        if original != cleaned:
            cleaned_count += 1
    
    print(f"✓ Cleaned {cleaned_count}/{len(data)} entries (some were already clean)")
    
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved cleaned data to {output_path}")
    print(f"Final dataset: {len(data)} entries with 'cleaned_model_output' field")


if __name__ == "__main__":
    main()
