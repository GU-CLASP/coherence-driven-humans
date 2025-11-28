#!/usr/bin/env python3
"""Collect model outputs from parquet files and human CSV files.

Example usage:
    python collect_data.py \
        --qwen3vl-out ../models/qwen3vl/out-qwen3vl-60stories/ \
        --internvl3-out ../models/internvl3/out-internvl3-60stories/ \
        --llama4-out ../models/llama4scout/out-llama4scout-60stories/ \
        --gpt4o-out ../models/gpt/out-gpt4o-60stories/ \
        --claude45-out ../models/claude/out-claude45-60stories/ \
        --human-large-csv ../notebooks/collected_60.csv \
        --human-original-csv ../data/vwp-acl2025-subset.csv \
        --output-json ../data/post-processing/collected_outputs.json
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--qwen3vl-out", required=True, help="qwen3vl output directory (e.g., ../models/qwen3vl/out-qwen3vl-60stories/)")
    parser.add_argument("--internvl3-out", required=True, help="internvl3 output directory (e.g., ../models/internvl3/out-internvl3-60stories/)")
    parser.add_argument("--llama4-out", required=True, help="llama4scout output directory (e.g., ../models/llama4scout/out-llama4scout-60stories/)")
    parser.add_argument("--gpt4o-out", required=True, help="gpt4o output directory (e.g., ../models/gpt/out-gpt4o-60stories/)")
    parser.add_argument("--claude45-out", required=True, help="claude45 output directory (e.g., ../models/claude/out-claude45-60stories/)")
    parser.add_argument("--human-large-csv", required=True, help="Human large prompt CSV (e.g., ../notebooks/collected_60.csv)")
    parser.add_argument("--human-original-csv", required=True, help="Human original CSV (e.g., ../data/vwp-acl2025-subset.csv)")
    parser.add_argument("--output-json", required=True, help="Output JSON file (e.g., ../data/post-processing/collected_outputs.json)")
    return parser.parse_args()


def create_story_images_ground_truth(model_configs, seeds_large):
    """Create ground-truth dictionary for num_story_images from open source models."""
    story_images_dict = {}
    
    print("Creating ground-truth dictionary for num_story_images...")
    
    for model_type, base_path in model_configs.items():
        if model_type in ['gpt4o', 'claude45']:
            continue  # Skip closed-source models
            
        print(f"Processing {model_type} from {base_path}")
        
        # Check original prompt outputs
        original_path = os.path.join(base_path, 'prompt-original-outputs')
        if os.path.exists(original_path):
            parquet_files = glob.glob(os.path.join(original_path, '*.parquet'))
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)
                    for _, row in df.iterrows():
                        story_id = int(row['story_id'])
                        num_images = int(row['num_story_images'])
                        if story_id not in story_images_dict:
                            story_images_dict[story_id] = num_images
                except Exception as e:
                    print(f"    Error processing {file_path}: {e}")
        
        # Check large prompt outputs
        large_path = os.path.join(base_path, 'prompt-large-outputs')
        if os.path.exists(large_path):
            for seed in seeds_large:
                seed_path = os.path.join(large_path, f'seed-{seed}')
                if os.path.exists(seed_path):
                    parquet_files = glob.glob(os.path.join(seed_path, '*.parquet'))
                    for file_path in parquet_files:
                        try:
                            df = pd.read_parquet(file_path)
                            for _, row in df.iterrows():
                                story_id = int(row['story_id'])
                                num_images = int(row['num_story_images'])
                                if story_id not in story_images_dict:
                                    story_images_dict[story_id] = num_images
                        except Exception as e:
                            print(f"    Error processing {file_path}: {e}")
    
    print(f"Ground-truth dictionary created with {len(story_images_dict)} stories\n")
    return story_images_dict


def collect_all_data(args):
    """Collect data from all models and human annotations."""
    seeds_large = ['42', '43', '44']
    all_data = []
    
    model_configs = {
        'qwen3vl': args.qwen3vl_out,
        'internvl3': args.internvl3_out,
        'llama4scout': args.llama4_out,
        'gpt4o': args.gpt4o_out,
        'claude45': args.claude45_out
    }
    
    ground_truth_images = create_story_images_ground_truth(model_configs, seeds_large)
    
    # Get target story_ids from human large outputs
    print("Determining target story_ids from human large outputs...")
    target_story_ids = set()
    try:
        human_large_df = pd.read_csv(args.human_large_csv)
        target_story_ids = set(human_large_df['story_id'].unique())
        print(f"  Found {len(target_story_ids)} unique story_ids\n")
    except Exception as e:
        print(f"Error reading human large outputs: {e}")
        return []
    
    # Process AI model outputs
    for model_type, base_path in model_configs.items():
        print(f"Processing {model_type} from {base_path}")
        
        # Process original prompt outputs
        original_path = os.path.join(base_path, 'prompt-original-outputs')
        if os.path.exists(original_path):
            parquet_files = glob.glob(os.path.join(original_path, '*.parquet'))
            print(f"  Found {len(parquet_files)} original prompt files")
            
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)
                    for _, row in df.iterrows():
                        if int(row['story_id']) in target_story_ids:
                            story_id = int(row['story_id'])
                            num_images = ground_truth_images.get(story_id,
                                       int(row['num_story_images']) if 'num_story_images' in row else None)
                            
                            all_data.append({
                                'story_id': story_id,
                                'seed': int(row['seed']),
                                'model_output': row['model_output'],
                                'model_type': model_type,
                                'prompt_type': 'original',
                                'num_story_images': num_images
                            })
                except Exception as e:
                    print(f"    Error processing {file_path}: {e}")
        
        # Process large prompt outputs
        large_path = os.path.join(base_path, 'prompt-large-outputs')
        if os.path.exists(large_path):
            for seed in seeds_large:
                seed_path = os.path.join(large_path, f'seed-{seed}')
                if os.path.exists(seed_path):
                    parquet_files = glob.glob(os.path.join(seed_path, '*.parquet'))
                    print(f"  Found {len(parquet_files)} large prompt files for seed {seed}")
                    
                    for file_path in parquet_files:
                        try:
                            df = pd.read_parquet(file_path)
                            for _, row in df.iterrows():
                                if int(row['story_id']) in target_story_ids:
                                    story_id = int(row['story_id'])
                                    num_images = ground_truth_images.get(story_id,
                                               int(row['num_story_images']) if 'num_story_images' in row else None)
                                    
                                    all_data.append({
                                        'story_id': story_id,
                                        'seed': int(row['seed']),
                                        'model_output': row['model_output'],
                                        'model_type': model_type,
                                        'prompt_type': 'large',
                                        'num_story_images': num_images
                                    })
                        except Exception as e:
                            print(f"    Error processing {file_path}: {e}")
    
    # Process human annotations - original prompts
    print("Processing human original prompt annotations...")
    try:
        human_original_df = pd.read_csv(args.human_original_csv)
        human_original_filtered = human_original_df[human_original_df['story_id'].isin(target_story_ids)]
        print(f"  Filtered to {len(human_original_filtered)} entries matching target story_ids")
        
        for _, row in human_original_filtered.iterrows():
            story_id = int(row['story_id'])
            num_images = ground_truth_images.get(story_id, None)
            if num_images is None and 'image_count' in row and pd.notna(row['image_count']):
                num_images = int(row['image_count'])
            
            # Use sep_story column which already has [SENT] separators, convert to [SEP]
            if 'sep_story' in row and pd.notna(row['sep_story']) and str(row['sep_story']).strip():
                story_text = str(row['sep_story']).strip().replace(' [SENT] ', ' [SEP] ')
                all_data.append({
                    'story_id': story_id,
                    'seed': 42,
                    'model_output': story_text,
                    'model_type': 'human',
                    'prompt_type': 'original',
                    'num_story_images': num_images
                })
    except Exception as e:
        print(f"Error processing human original prompts: {e}")
    
    # Process human annotations - large prompts
    print("Processing human large prompt annotations...")
    try:
        human_large_df = pd.read_csv(args.human_large_csv)
        random.seed(42)
        
        for story_id, group in human_large_df.groupby('story_id'):
            num_images = ground_truth_images.get(story_id, None)
            
            stories = []
            for _, story_row in group.iterrows():
                story_parts = []
                for i in range(10):
                    text_col = f'text{i}'
                    if text_col in story_row and pd.notna(story_row[text_col]) and str(story_row[text_col]).strip():
                        story_parts.append(str(story_row[text_col]).strip())
                
                if story_parts:
                    stories.append(' [SEP] '.join(story_parts))
            
            if len(stories) >= 3:
                available_seeds = seeds_large.copy()
                random.shuffle(available_seeds)
                
                for i, story in enumerate(stories[:3]):
                    all_data.append({
                        'story_id': int(story_id),
                        'seed': int(available_seeds[i]),
                        'model_output': story,
                        'model_type': 'human',
                        'prompt_type': 'large',
                        'num_story_images': num_images
                    })
            else:
                print(f"  Warning: Only found {len(stories)} stories for story_id {story_id}, expected 3")
    except Exception as e:
        print(f"Error processing human large prompts: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total entries collected: {len(all_data)}")
    
    return all_data


def main():
    args = parse_args()
    collected_data = collect_all_data(args)
    
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(collected_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(collected_data)} entries to {output_path}")


if __name__ == "__main__":
    main()
