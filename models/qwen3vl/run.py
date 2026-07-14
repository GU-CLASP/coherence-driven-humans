import os
import pandas as pd
from tqdm import tqdm
import argparse
import time
import base64
import glob
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from nltk.tokenize import wordpunct_tokenize

MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Thinking"

# Will be initialized in main() with the server URL from command line
client = None

# Local image paths
IMAGES_DIR = "/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/data/sampled_60/images"
CHARACTERS_DIR = "/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/data/sampled_60/characters"

def initialize_client(server_url):
    """Initialize the OpenAI client with the given server URL."""
    global client
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{server_url}/v1",
        timeout=3600
    )

def local_image_to_data_url(image_path):
    """Convert a local image file to a base64 data URL."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Determine mime type from extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    
    return f"data:{mime_type};base64,{image_data}"

def get_story_images_for_story_id(story_id):
    """Get all story images for a given story_id, ordered by image number."""
    pattern = os.path.join(IMAGES_DIR, f"{story_id}_img*.jpg")
    images = sorted(glob.glob(pattern), key=lambda x: int(x.split('_img')[1].split('.')[0]))
    return images

def get_character_images_for_story_id(story_id):
    """Get all character images for a given story_id, ordered by character number."""
    pattern = os.path.join(CHARACTERS_DIR, f"{story_id}_char*.jpg")
    images = sorted(glob.glob(pattern), key=lambda x: int(x.split('_char')[1].split('.')[0]))
    return images

def count_words(text):
    """Count words in text using NLTK tokenization."""
    if pd.isna(text):
        return 0

    tokens = wordpunct_tokenize(str(text))
    return sum(1 for token in tokens if re.search(r"\w", token))

def strip_story_markers(text):
    """Remove structural markers that should not count as words."""
    return re.sub(r"\[(?:SEP|SENT)\]", " ", str(text))

def send_prompt_with_images_openai(story_images, character_images, instruction_text, seed, template_name):
    """Send prompt with local images (converted to base64) to the model."""
    content = []

    # Add story images
    for image_path in story_images:
        data_url = local_image_to_data_url(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url}
        })
    
    # Add character images if available
    if character_images:
        for image_path in character_images:
            data_url = local_image_to_data_url(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })

    content.append({
        "type": "text",
        "text": instruction_text
    })
    
    # Determine system message based on template
    if template_name.startswith("large"):
        system_message = (
            "You are a helpful assistant and an experienced expert crowdworker. "
            "You are qualified to perform the following task. The title of the task you are working on is: "
            "\"Help us bridge the gap between AI and humans in telling stories about movies!\" "
            "The task description is as follows: we are a group of researchers working with large language models, "
            "and we ask for your help in collecting stories based on the images provided. The data you submit will "
            "be used to build and improve AI models that understand how to generate stories about movies just as you do! "
            "We're very excited to have you join our experiment! Please carefully read the instructions. "
            "You must follow all instructions in order to be eligible for payment."
        )
    else:
        system_message = (
            "You are a helpful assistant and an experienced expert crowdworker. "
            "You are qualified to perform the following task."
        )
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_message,
                    }
                ],
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
        seed=seed
    )
    
    return response.choices[0].message.content



def load_template(template_dir, template_name):
    template_path = os.path.join(template_dir, f'prompt-{template_name}.txt')
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    with open(template_path, 'r', encoding='utf-8') as f:
        template_text = f.read()
    return template_text

def get_instruction_text(
    template_dir,
    base_template_name,
    story_images,
    character_images=None,
    character_names=None,
    target_words=None
):
    if character_images and len(character_images) > 0:
        template_name = f"{base_template_name}-w-names"
    else:
        template_name = f"{base_template_name}-wo-names"

    template = load_template(template_dir, template_name)

    fill_values = {
        'story_images': [os.path.basename(p) for p in story_images],
        'num_story_images': len(story_images),
        'target_words': target_words if target_words is not None else ""
    }

    if character_images:
        fill_values['character_images'] = [os.path.basename(p) for p in character_images]
        fill_values['num_character_images'] = len(character_images)
        fill_values['character_images_text'] = (
            '1 character image' if len(character_images) == 1 else f"{len(character_images)} character images"
        )
    else:
        fill_values['character_images'] = []
        fill_values['num_character_images'] = 0
        fill_values['character_images_text'] = 'no character images'

    print('CHARACTER NAMES', character_names)
    if character_names:
        fill_values['character_names'] = character_names
        fill_values['character_names_text'] = ', '.join(character_names)
    else:
        fill_values['character_names'] = []
        fill_values['character_names_text'] = 'no character names'

    filled_instruction = template.format(**fill_values)
    return filled_instruction

def extract_story_ids_from_csv(csv_file):
    """Extract unique story IDs from the CSV file."""
    df = pd.read_csv(csv_file)
    story_ids = sorted(df['story_id'].unique().tolist())
    return df, story_ids

def load_jsonl_dataframe(jsonl_path):
    """Load a JSONL file into a DataFrame."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)

def load_json_dataframe(json_path):
    """Load a JSON file containing a list of records into a DataFrame."""
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {json_path}, got {type(records).__name__}")
    return pd.DataFrame(records)

def combine_text_fields(row, text_fields):
    """Combine multiple text fields into a single story string."""
    parts = []
    for field in text_fields:
        if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
            parts.append(str(row[field]).strip())
    return " [SEP] ".join(parts)

def compute_target_words_by_story_id(target_source_file, target_source_format, text_column=None):
    """Compute per-story target word counts from original or large human annotations."""
    if target_source_format == "jsonl":
        df = load_jsonl_dataframe(target_source_file)
        if text_column is None:
            text_column = "story"
        if text_column not in df.columns:
            raise ValueError(
                f"Target text column '{text_column}' not found in {target_source_file}. "
                f"Available columns: {list(df.columns)}"
            )
        target_df = df[["story_id", text_column]].copy()
        target_df["word_count"] = target_df[text_column].apply(count_words)
        grouped = target_df.groupby("story_id", as_index=False)["word_count"].mean()
        grouped["target_words"] = grouped["word_count"].round().astype(int)
        return dict(zip(grouped["story_id"], grouped["target_words"]))

    if target_source_format == "csv":
        df = pd.read_csv(target_source_file)
        text_fields = [f"text{i}" for i in range(10)]
        if text_column:
            text_fields = [field.strip() for field in text_column.split(",") if field.strip()]
        missing_fields = [field for field in text_fields if field not in df.columns]
        if missing_fields:
            raise ValueError(
                f"Target text fields {missing_fields} not found in {target_source_file}. "
                f"Available columns: {list(df.columns)}"
            )

        target_df = df[["story_id"] + text_fields].copy()
        target_df["combined_text"] = target_df.apply(lambda row: combine_text_fields(row, text_fields), axis=1)
        target_df["word_count"] = target_df["combined_text"].apply(count_words)
        grouped = target_df.groupby("story_id", as_index=False)["word_count"].median()
        grouped["target_words"] = grouped["word_count"].round().astype(int)
        return dict(zip(grouped["story_id"], grouped["target_words"]))

    if target_source_format == "cleaned_outputs":
        df = load_json_dataframe(target_source_file)
        required_columns = {"story_id", "model_type", "prompt_type", "cleaned_model_output"}
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(
                f"Target source {target_source_file} is missing required columns {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

        target_df = df[
            (df["model_type"] == "human")
            & (df["prompt_type"] == "large")
        ][["story_id", "cleaned_model_output"]].copy()

        if target_df.empty:
            raise ValueError(
                f"No rows with model_type='human' and prompt_type='large' found in {target_source_file}"
            )

        target_df["word_count"] = target_df["cleaned_model_output"].apply(
            lambda text: count_words(strip_story_markers(text))
        )
        grouped = target_df.groupby("story_id", as_index=False)["word_count"].mean()
        grouped["target_words"] = grouped["word_count"].round().astype(int)
        return dict(zip(grouped["story_id"], grouped["target_words"]))

    raise ValueError(f"Unknown target_source_format: {target_source_format}")

def default_target_source_for_template(template_name):
    """Pick the correct human source file for a prompt template."""
    repo_root = "/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans"
    if template_name == "large-upper-bound":
        return os.path.join(repo_root, "data", "post-processing", "cleaned_outputs.json"), "cleaned_outputs"
    if "large" in template_name:
        return os.path.join(repo_root, "notebooks", "collected_60.csv"), "csv"
    return os.path.join(repo_root, "data", "sampled_60", "sampled_60_stories.json"), "jsonl"

def load_character_names_dataframe():
    """Load the character names from the VWP dataset CSV."""
    vwp_csv_path = "/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/data/vwp-acl2025-subset.csv"
    return pd.read_csv(vwp_csv_path)

def get_character_names_for_story_id(vwp_df, story_id):
    """Get character names for a given story_id from the VWP dataframe."""
    row = vwp_df[vwp_df['story_id'] == story_id].iloc[0] if len(vwp_df[vwp_df['story_id'] == story_id]) > 0 else None
    
    if row is None:
        return None
    
    character_names = []
    for i in range(5):  # Check up to 5 characters
        char_col = f'char{i}'
        if char_col in row.index:
            char_name = row[char_col]
            # Only add if it's a valid name (not empty, not NaN)
            if pd.notna(char_name) and char_name != '' and char_name != '{}':
                character_names.append(char_name)
    
    return character_names if character_names else None

def process_story(story_id, output_dir, prompt_dir, prompt_name, seed, vwp_df, target_words_map):
    """Process a single story by querying the model with its images."""
    try:
        # Determine output path first
        if prompt_name.startswith("large"):
            output_subdir = os.path.join(output_dir, f"seed-{seed}")
        else:
            output_subdir = output_dir
        
        output_file = os.path.join(output_subdir, f"{story_id}.parquet")
        
        # Check if output already exists
        if os.path.exists(output_file):
            print(f"Output file already exists for story_id {story_id}, skipping...")
            return True
        
        #if story_id != 5201:
        print(story_id)
        # Get story and character images
        story_images = get_story_images_for_story_id(story_id)
        character_images = get_character_images_for_story_id(story_id)
        character_names = get_character_names_for_story_id(vwp_df, story_id)
        target_words = target_words_map.get(story_id)

        if "target" in prompt_name and target_words is None:
            print(f"Warning: No target_words found for story_id {story_id}, skipping...")
            return False
        
        if not story_images:
            print(f"Warning: No story images found for story_id {story_id}")
            return False
        
        # Generate instruction text
        instruction_text = get_instruction_text(
            prompt_dir,
            base_template_name=prompt_name,
            story_images=story_images,
            character_images=character_images if character_images else None,
            character_names=character_names,
            target_words=target_words
        )
        
        # Query the model
        start_time = time.time()


        model_output = send_prompt_with_images_openai(
            story_images=story_images,
            character_images=character_images if character_images else None,
            instruction_text=instruction_text,
            seed=seed,
            template_name=prompt_name
        )
        elapsed_time = time.time() - start_time
        
        print(f"[{elapsed_time:.2f}s] Story {story_id}: {model_output}")
        
        # Save output
        os.makedirs(output_subdir, exist_ok=True)
        
        result_df = pd.DataFrame([{
            "story_id": story_id,
            "num_story_images": len(story_images),
            "num_character_images": len(character_images),
            "target_words": target_words,
            "instruction_text": instruction_text,
            "model_output": model_output,
            "seed": seed,
            "elapsed_time": elapsed_time
        }])
        result_df.to_parquet(output_file, index=False)
        
        return True
            
    except Exception as e:
        print(f"Error processing story_id {story_id}: {e}")
        return False

def run(story_ids, output_dir, prompt_dir, prompt_name, seed, vwp_df, target_words_map, concurrency=1):
    """Process all stories and save outputs.

    When concurrency > 1, stories are dispatched to the server concurrently via a
    thread pool. The OpenAI client is thread-safe and each story writes its own
    parquet file, so this is safe. The server batches the concurrent requests
    (bounded by its --max-num-seqs).
    """
    os.makedirs(output_dir, exist_ok=True)

    successful = 0
    failed = 0

    if concurrency <= 1:
        for story_id in tqdm(story_ids, total=len(story_ids)):
            if process_story(story_id, output_dir, prompt_dir, prompt_name, seed, vwp_df, target_words_map):
                successful += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    process_story, story_id, output_dir, prompt_dir, prompt_name,
                    seed, vwp_df, target_words_map,
                )
                for story_id in story_ids
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                if future.result():
                    successful += 1
                else:
                    failed += 1

    print(f"\nProcessing complete: {successful} successful, {failed} failed")


def main(args):
    # Initialize client with the provided server URL
    initialize_client(args.server_url)
    print(f"Initialized client with server: http://{args.server_url}/v1")

    template_dir = args.template_dir
    template_name = args.template_name

    target_source_file = args.target_source_file
    target_source_format = args.target_source_format
    target_text_column = args.target_text_column

    if target_source_file is None:
        target_source_file, target_source_format = default_target_source_for_template(template_name)
    elif target_source_format == "auto":
        lower_path = target_source_file.lower()
        if lower_path.endswith((".jsonl", ".json")):
            target_source_format = "jsonl"
        else:
            target_source_format = "csv"

    target_words_map = compute_target_words_by_story_id(
        target_source_file=target_source_file,
        target_source_format=target_source_format,
        text_column=target_text_column,
    )
    print(
        f"Computed target_words for {len(target_words_map)} stories from "
        f"{target_source_file} ({target_source_format})"
    )

    # Determine which stories to process. The CSV is optional: when it is not
    # provided (e.g. the original/short prompt), take the story IDs from the same
    # target source that provides the human word counts.
    if args.csv_file:
        story_df, story_ids = extract_story_ids_from_csv(args.csv_file)
        print(f"Found {len(story_ids)} unique stories in CSV")
    else:
        story_ids = sorted(target_words_map.keys())
        print(f"No --csv_file provided; using {len(story_ids)} story_ids from {target_source_file}")
    
    # Load character names from VWP dataset
    vwp_df = load_character_names_dataframe()
    print(f"Loaded {len(vwp_df)} rows from VWP dataset")
    
    output_dir = os.path.join(args.output_dir, f'prompt-{template_name}-outputs')
    os.makedirs(output_dir, exist_ok=True)

    if template_name in ['original', 'medium', 'original-target']:
        print(f"Running with template: {template_name}, seed 42")
        run(story_ids, output_dir, template_dir, template_name, seed=42, vwp_df=vwp_df, target_words_map=target_words_map, concurrency=args.concurrency)
    elif template_name in ['large', 'large-target', 'large-upper-bound']:
        print(f"Running with template: {template_name}, all 60 stories with seeds 42, 43, 44")
        for seed in [42, 43, 44]:
            print(f"\n=== Processing with seed {seed} ===")
            run(story_ids, output_dir, template_dir, template_name, seed=seed, vwp_df=vwp_df, target_words_map=target_words_map, concurrency=args.concurrency)
    else:
        raise ValueError(f"Unknown template name: {template_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stories using Qwen3 VL model")
    parser.add_argument("--csv_file", type=str, default=None, help="Optional CSV with a story_id column. If omitted, story IDs are taken from the target source (e.g. sampled_60_stories.json for original prompts).")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save outputs")
    parser.add_argument("--template_dir", type=str, default="/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans/data/prompts", help="Directory containing prompt templates")
    parser.add_argument(
        "--template_name",
        type=str,
        required=True,
        choices=["original", "medium", "large", "original-target", "large-target", "large-upper-bound"],
        help="Template name"
    )
    parser.add_argument(
        "--target_source_file",
        type=str,
        default=None,
        help="Optional source file for target word counts. Defaults to sampled_60_stories.json for original prompts and collected_60.csv for large prompts"
    )
    parser.add_argument(
        "--target_source_format",
        type=str,
        default="auto",
        choices=["auto", "csv", "jsonl", "cleaned_outputs"],
        help="Format of the target source file"
    )
    parser.add_argument(
        "--target_text_column",
        type=str,
        default=None,
        help="Text column for JSONL original stories, or comma-separated text fields for CSV large stories"
    )
    parser.add_argument("--server_url", type=str, required=True, help="Server URL in format 'hostname:port' (e.g., '10.21.30.119:47246')")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of stories to send concurrently. Should be <= the server's --max-num-seqs for best effect.")

    args = parser.parse_args()
    main(args)


