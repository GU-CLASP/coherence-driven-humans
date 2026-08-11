import argparse
import base64
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

MODEL_NAME = "OpenGVLab/InternVL3-78B-Instruct"

client = None

REPO_ROOT = "/nobackup/proj/disk/naiss2024-6-297/shared/coherence-driven-humans"
IMAGES_DIR = os.path.join(REPO_ROOT, "data", "sampled_60", "images")
CHARACTERS_DIR = os.path.join(REPO_ROOT, "data", "sampled_60", "characters")


def initialize_client(server_url):
    global client
    client = OpenAI(api_key="EMPTY", base_url=f"http://{server_url}/v1", timeout=3600)


def local_image_to_data_url(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime_type};base64,{image_data}"


def get_story_images_for_story_id(story_id):
    pattern = os.path.join(IMAGES_DIR, f"{story_id}_img*.jpg")
    return sorted(glob.glob(pattern), key=lambda x: int(x.split("_img")[1].split(".")[0]))


def get_character_images_for_story_id(story_id):
    pattern = os.path.join(CHARACTERS_DIR, f"{story_id}_char*.jpg")
    return sorted(glob.glob(pattern), key=lambda x: int(x.split("_char")[1].split(".")[0]))


def send_prompt_with_images_openai(story_images, character_images, instruction_text, seed, template_name, max_tokens=4096):
    content = []

    for image_path in story_images:
        data_url = local_image_to_data_url(image_path)
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    if character_images:
        for image_path in character_images:
            data_url = local_image_to_data_url(image_path)
            content.append({"type": "image_url", "image_url": {"url": data_url}})

    if template_name == "large":
        system_message = (
            "You are a helpful assistant and an experienced expert crowdworker. "
            "You are qualified to perform the following task. The title of the task you are working on is: "
            '"Help us bridge the gap between AI and humans in telling stories about movies!" '
            "The task description is as follows: we are a group of researchers working with large language models, "
            "and we ask for your help in collecting stories based on the images provided. The data you submit will "
            "be used to build and improve AI models that understand how to generate stories about movies just as you do! "
            "We're very excited to have you join our experiment! Please carefully read the instructions. "
            "You must follow all instructions in order to be eligible for payment.\n\n"
        )
    else:
        system_message = (
            "You are a helpful assistant and an experienced expert crowdworker. "
            "You are qualified to perform the following task.\n\n"
        )

    full_text = system_message + instruction_text
    content.append({"type": "text", "text": full_text})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_tokens,
        seed=seed,
    )

    return response.choices[0].message.content


def load_template(template_dir, template_name):
    template_path = os.path.join(template_dir, f"prompt-{template_name}.txt")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def get_instruction_text(template_dir, base_template_name, story_images, character_images=None, character_names=None, target_words=None):
    if character_images and len(character_images) > 0:
        template_name = f"{base_template_name}-w-names"
    else:
        template_name = f"{base_template_name}-wo-names"

    template = load_template(template_dir, template_name)

    fill_values = {
        "story_images": [os.path.basename(p) for p in story_images],
        "num_story_images": len(story_images),
        "target_words": target_words if target_words is not None else "",
    }

    if character_images:
        fill_values["character_images"] = [os.path.basename(p) for p in character_images]
        fill_values["num_character_images"] = len(character_images)
        fill_values["character_images_text"] = (
            "1 character image" if len(character_images) == 1 else f"{len(character_images)} character images"
        )
    else:
        fill_values["character_images"] = []
        fill_values["num_character_images"] = 0
        fill_values["character_images_text"] = "no character images"

    if character_names:
        fill_values["character_names"] = character_names
        fill_values["character_names_text"] = ", ".join(character_names)
    else:
        fill_values["character_names"] = []
        fill_values["character_names_text"] = "no character names"

    return template.format(**fill_values)


def extract_story_ids_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    story_ids = sorted(df["story_id"].unique().tolist())
    return df, story_ids


def count_words(text):
    if pd.isna(text):
        return 0
    return len(re.findall(r"\b\w+\b", str(text)))


def strip_story_markers(text):
    return re.sub(r"\[(?:SEP|SENT)\]", " ", str(text))


def load_jsonl_dataframe(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def combine_text_fields(row, text_fields):
    parts = []
    for field in text_fields:
        if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
            parts.append(str(row[field]).strip())
    return " [SEP] ".join(parts)


def compute_target_words_by_story_id(target_source_file, target_source_format, text_column=None):
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
        with open(target_source_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)

        required_columns = {"story_id", "model_type", "prompt_type", "cleaned_model_output"}
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(
                f"Target source {target_source_file} is missing required columns {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

        target_df = df[(df["model_type"] == "human") & (df["prompt_type"] == "large")][
            ["story_id", "cleaned_model_output"]
        ].copy()

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
    if "large" in template_name:
        return os.path.join(REPO_ROOT, "collected_60.csv"), "csv"
    return os.path.join(REPO_ROOT, "data", "sampled_60", "sampled_60_stories.json"), "jsonl"


def load_character_names_dataframe():
    vwp_csv_path = os.path.join(REPO_ROOT, "data", "vwp-acl2025-subset.csv")
    return pd.read_csv(vwp_csv_path)


def get_character_names_for_story_id(vwp_df, story_id):
    row = vwp_df[vwp_df["story_id"] == story_id].iloc[0] if len(vwp_df[vwp_df["story_id"] == story_id]) > 0 else None

    if row is None:
        return None

    character_names = []
    for i in range(5):
        char_col = f"char{i}"
        if char_col in row.index:
            char_name = row[char_col]
            if pd.notna(char_name) and char_name != "" and char_name != "{}":
                character_names.append(char_name)

    return character_names if character_names else None


def process_story(story_id, output_dir, prompt_dir, prompt_name, seed, vwp_df, target_words_map, max_tokens):
    try:
        if prompt_name.startswith("large"):
            output_subdir = os.path.join(output_dir, f"seed-{seed}")
        else:
            output_subdir = output_dir

        output_file = os.path.join(output_subdir, f"{story_id}.parquet")

        if os.path.exists(output_file):
            print(f"Output file already exists for story_id {story_id}, skipping...")
            return True

        print(story_id)
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

        instruction_text = get_instruction_text(
            prompt_dir,
            base_template_name=prompt_name,
            story_images=story_images,
            character_images=character_images if character_images else None,
            character_names=character_names,
            target_words=target_words,
        )

        start_time = time.time()

        model_output = send_prompt_with_images_openai(
            story_images=story_images,
            character_images=character_images if character_images else None,
            instruction_text=instruction_text,
            seed=seed,
            template_name=prompt_name,
            max_tokens=max_tokens,
        )
        elapsed_time = time.time() - start_time

        print(f"[{elapsed_time:.2f}s] Story {story_id}: {model_output}")

        os.makedirs(output_subdir, exist_ok=True)

        result_df = pd.DataFrame(
            [
                {
                    "story_id": story_id,
                    "num_story_images": len(story_images),
                    "num_character_images": len(character_images),
                    "target_words": target_words,
                    "instruction_text": instruction_text,
                    "model_output": model_output,
                    "seed": seed,
                    "elapsed_time": elapsed_time,
                }
            ]
        )
        result_df.to_parquet(output_file, index=False)

        return True

    except Exception as e:
        print(f"Error processing story_id {story_id}: {e}")
        return False


def run(story_ids, output_dir, prompt_dir, prompt_name, seed, vwp_df, target_words_map, max_tokens, concurrency=1):
    os.makedirs(output_dir, exist_ok=True)

    successful = 0
    failed = 0

    if concurrency <= 1:
        for story_id in tqdm(story_ids, total=len(story_ids)):
            if process_story(story_id, output_dir, prompt_dir, prompt_name, seed, vwp_df, target_words_map, max_tokens):
                successful += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    process_story,
                    story_id,
                    output_dir,
                    prompt_dir,
                    prompt_name,
                    seed,
                    vwp_df,
                    target_words_map,
                    max_tokens,
                )
                for story_id in story_ids
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                if future.result():
                    successful += 1
                else:
                    failed += 1

    print(f"\nProcessing complete: {successful} successful, {failed} failed")
    return successful, failed


def main(args):
    initialize_client(args.server_url)
    print(f"Initialized client with server: http://{args.server_url}/v1")

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

    if args.csv_file:
        _, story_ids = extract_story_ids_from_csv(args.csv_file)
        print(f"Found {len(story_ids)} unique stories in CSV")
    else:
        story_ids = sorted(target_words_map.keys())
        print(f"No --csv_file provided; using {len(story_ids)} story_ids from {target_source_file}")

    vwp_df = load_character_names_dataframe()
    print(f"Loaded {len(vwp_df)} rows from VWP dataset")

    template_dir = args.template_dir
    output_dir = os.path.join(args.output_dir, f"prompt-{template_name}-outputs")
    os.makedirs(output_dir, exist_ok=True)

    if template_name in ["original", "medium", "original-target"]:
        print(f"Running with template: {template_name}, seed 42")
        _, failed = run(
            story_ids,
            output_dir,
            template_dir,
            template_name,
            seed=42,
            vwp_df=vwp_df,
            target_words_map=target_words_map,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
        return 1 if failed else 0

    if template_name in ["large", "large-target", "large-upper-bound"]:
        print(f"Running with template: {template_name}, all 60 stories with seeds 42, 43, 44")
        total_failed = 0
        for seed in [42, 43, 44]:
            print(f"\n=== Processing with seed {seed} ===")
            _, failed = run(
                story_ids,
                output_dir,
                template_dir,
                template_name,
                seed=seed,
                vwp_df=vwp_df,
                target_words_map=target_words_map,
                max_tokens=args.max_tokens,
                concurrency=args.concurrency,
            )
            total_failed += failed
        return 1 if total_failed else 0

    raise ValueError(f"Unknown template name: {template_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stories using InternVL3 model")
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="Optional CSV with a story_id column. If omitted, story IDs are taken from the target source.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save outputs")
    parser.add_argument(
        "--template_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "prompts"),
        help="Directory containing prompt templates",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        required=True,
        choices=["original", "medium", "large", "original-target", "large-target", "large-upper-bound"],
        help="Template name",
    )
    parser.add_argument(
        "--target_source_file",
        type=str,
        default=None,
        help=(
            "Optional source file for target word counts. Defaults to sampled_60_stories.json "
            "for original prompts and collected_60.csv for large prompts"
        ),
    )
    parser.add_argument(
        "--target_source_format",
        type=str,
        default="auto",
        choices=["auto", "csv", "jsonl", "cleaned_outputs"],
        help="Format of the target source file",
    )
    parser.add_argument(
        "--target_text_column",
        type=str,
        default=None,
        help="Text column for JSONL original stories, or comma-separated text fields for CSV large stories",
    )
    parser.add_argument("--server_url", type=str, required=True, help="Server URL in format hostname:port")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of stories to send concurrently. Should be <= the server's --max-num-seqs.",
    )
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum generated tokens per story request")

    args = parser.parse_args()
    raise SystemExit(main(args))
