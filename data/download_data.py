#!/usr/bin/env python3
"""Sample 20 stories, then 40 more (total 60), save all to one folder.
The images are downloaded from the subset used in GEM paper, https://aclanthology.org/2025.gem-1.67/
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-file", required=True, help="vwp-acl2025-subset.csv path")
    parser.add_argument("--output-dir", required=True,
                        help="Output folder (will contain images/, characters/, and JSON files)")
    return parser.parse_args()


def download_image(url: str, save_path: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=10) as response:
            response.raise_for_status()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    handle.write(chunk)
        return True
    except requests.RequestException as exc:
        print(f"Failed to download {url} -> {save_path}: {exc}")
        return False


def is_valid_url(value):
    """Check if value is a valid URL (not None, NaN, empty, or '{}')."""
    if value is None:
        return False
    if pd.isna(value):
        return False
    value_str = str(value).strip()
    if value_str == '' or value_str.lower() == 'nan' or value_str == '{}':
        return False
    return True


def download_stories(df: pd.DataFrame,
                    images_dir: Path,
                    characters_dir: Path,
                    desc: str) -> pd.DataFrame:
    """Download images for a subset of stories."""
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(characters_dir, exist_ok=True)

    # there are 10 images max and 5 character images max
    image_columns = [f"link{i}" for i in range(10)]
    char_url_columns = [f"char{i}_url" for i in range(5)]
    char_name_columns = [f"char{i}" for i in range(5)]

    df["images"] = [[] for _ in range(len(df))]
    df["characters"] = [[] for _ in range(len(df))]

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=desc):
        story_id = row["story_id"]
        image_paths = []
        # download all story images (up to 10 per story)
        for img_idx, link in enumerate(image_columns):
            if link not in df.columns:
                continue
            url = row[link]
            if is_valid_url(url):
                filename = f"{story_id}_img{img_idx}.jpg"
                save_path = Path(images_dir) / filename
                # if the image has already been downloaded, skip download
                if save_path.exists() or download_image(url, save_path):
                    image_paths.append(str(save_path))
        df.at[idx, "images"] = image_paths

        character_info = []
        # download all character images (up to 5 per story)
        for char_idx, (char_name_col, char_url_col) in enumerate(zip(char_name_columns, char_url_columns)):
            if char_name_col not in df.columns or char_url_col not in df.columns:
                continue
            char_name = row[char_name_col]
            char_url = row[char_url_col]
            if is_valid_url(char_url) and is_valid_url(char_name):
                filename = f"{story_id}_char{char_idx}.jpg"
                save_path = Path(characters_dir) / filename
                # if the character image has already been downloaded, skip download
                if save_path.exists() or download_image(char_url, save_path):
                    character_info.append({"image": str(save_path), "name": char_name})
        df.at[idx, "characters"] = character_info

    return df





def main() -> None:
    args = parse_args()
    csv_file = Path(args.csv_file)
    output_dir = Path(args.output_dir)
    
    images_dir = output_dir / "images"
    characters_dir = output_dir / "characters"
    
    df = pd.read_csv(csv_file)
    
    # sample first 20 stories
    print("Sampling first 20 stories...")
    subset_20 = df.dropna(subset=["text0"]).copy()
    subset_20 = subset_20.sample(n=20, random_state=20).reset_index(drop=True)
    
    # download images for first 20
    subset_20 = download_stories(subset_20, images_dir, characters_dir, "Downloading 20 stories")
    
    # save first 20
    output_dir.mkdir(parents=True, exist_ok=True)
    json_20 = output_dir / "sampled_20_stories.json"
    subset_20.to_json(json_20, orient="records", lines=True, force_ascii=False)
    print(f"Saved 20 stories to {json_20}")
    
    # get story IDs to exclude
    story_ids_20 = subset_20["story_id"].tolist()
    
    # sample extra 40 stories
    print("\nSampling extra 40 stories...")
    working = df.copy()
    if "split" in working.columns:
        working = working[working["split"] != "test"]
    working = working[~working["story_id"].isin(story_ids_20)]
    working = working.dropna(subset=["text0"])
    
    subset_40 = working.sample(n=40, random_state=42).reset_index(drop=True)
    
    # download images for extra 40
    subset_40 = download_stories(subset_40, images_dir, characters_dir, "Downloading 40 stories")
    
    # save extra 40
    json_40 = output_dir / "sampled_40_stories.json"
    subset_40.to_json(json_40, orient="records", lines=True, force_ascii=False)
    print(f"Saved 40 stories to {json_40}")
    
    # combine all 60 stories
    all_60 = pd.concat([subset_20, subset_40], ignore_index=True)
    json_60 = output_dir / "sampled_60_stories.json"
    all_60.to_json(json_60, orient="records", lines=True, force_ascii=False)
    print(f"\nSaved all 60 stories to {json_60}")
    print(f"Images saved to {images_dir}")
    print(f"Characters saved to {characters_dir}")


if __name__ == "__main__":
    main()
