#!/usr/bin/env python3
"""Prepare LinkAppend input files from cleaned model outputs.

This script converts cleaned story outputs into LinkAppend-compatible JSON format.
Each story is tokenized and formatted with sentence-level structure.

Example usage:
    python prepare_linkappend_inputs.py \
        --input-json ../data/post-processing/cleaned_outputs.json \
        --output-dir ../models/linkappend/data-in
"""

import argparse
import json
import os
from pathlib import Path
import nltk
from tqdm import tqdm

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-json", required=True,
                       help="Input cleaned JSON file (e.g., ../data/post-processing/cleaned_outputs.json)")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for LinkAppend inputs (e.g., ../models/linkappend/data-in)")
    return parser.parse_args()


def process_story_to_linkappend_format(story_id, text, doc_index, model_type, prompt_type, seed):
    """Convert a story text into LinkAppend JSON format.
    
    Args:
        story_id: The story ID
        text: The story text with [SEP] separators
        doc_index: Document index for unique ID
        model_type: Type of model (human, qwen3vl, etc.)
        prompt_type: Type of prompt (original, large)
        seed: Random seed used
        
    Returns:
        Dictionary in LinkAppend format
    """
    # Clean up text: remove leading/trailing [SEP], extra spaces
    text = text.strip()
    while text.startswith('[SEP]'):
        text = text[5:].strip()
    while text.endswith('[SEP]'):
        text = text[:-5].strip()
    
    # Split by [SEP] separator
    sentences = text.split(' [SEP] ')
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    
    # Additional cleanup: remove any remaining [SEP] at start/end of sentences
    cleaned_sentences = []
    for sent in sentences:
        sent = sent.strip()
        while sent.startswith('[SEP]'):
            sent = sent[5:].strip()
        while sent.endswith('[SEP]'):
            sent = sent[:-5].strip()
        if sent:
            cleaned_sentences.append(sent)
    
    # Process each sentence
    processed_sentences = []
    for i, sentence in enumerate(cleaned_sentences, 1):
        # Tokenize the sentence
        words = nltk.word_tokenize(sentence)
        tokens = [{'id': idx + 1, 'text': word} for idx, word in enumerate(words)]
        
        processed_sentences.append({
            'id': i,
            'speaker': None,
            'text': sentence,
            'tokens': tokens
        })
    
    # Create document ID
    doc_id = f'doc_{story_id}_{model_type}_{prompt_type}_seed{seed}_{doc_index}'
    
    return {
        'id': doc_id,
        'sentences': processed_sentences,
        'coref_chains': None
    }


def main():
    args = parse_args()
    
    print(f"Loading {args.input_json}...")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group by model/prompt/seed
    grouped_data = {}
    for entry in data:
        key = f"{entry['model_type']}_{entry['prompt_type']}_seed{entry['seed']}"
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(entry)

    print(f"Processing {len(data)} entries across {len(grouped_data)} combinations...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each group with progress bar
    for key, entries in tqdm(sorted(grouped_data.items()), desc="Processing groups"):
        linkappend_docs = []
        
        for idx, entry in enumerate(entries):
            doc = process_story_to_linkappend_format(
                story_id=entry['story_id'],
                text=entry['cleaned_model_output'],
                doc_index=idx,
                model_type=entry['model_type'],
                prompt_type=entry['prompt_type'],
                seed=entry['seed']
            )
            linkappend_docs.append(doc)
        
        output_file = output_dir / f"{key}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(linkappend_docs, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(grouped_data)} files to {output_dir}")


if __name__ == "__main__":
    main()
