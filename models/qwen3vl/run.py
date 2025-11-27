import os
import pandas as pd
from tqdm import tqdm
import argparse
import time
import base64
import glob
from openai import OpenAI

MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Thinking"

# Will be initialized in main() with the server URL from command line
client = None

# Local image paths
IMAGES_DIR = "/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/data/sampled_60/images"
CHARACTERS_DIR = "/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/data/sampled_60/characters"

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
    if template_name == "large":
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

def get_instruction_text(template_dir, base_template_name, story_images, character_images=None, character_names=None):
    if character_images and len(character_images) > 0:
        template_name = f"{base_template_name}-w-names"
    else:
        template_name = f"{base_template_name}-wo-names"

    template = load_template(template_dir, template_name)

    fill_values = {
        'story_images': [os.path.basename(p) for p in story_images],
        'num_story_images': len(story_images)
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

def load_character_names_dataframe():
    """Load the character names from the VWP dataset CSV."""
    vwp_csv_path = "/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/data/vwp-acl2025-subset.csv"
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

def process_story(story_id, output_dir, prompt_dir, prompt_name, seed, vwp_df):
    """Process a single story by querying the model with its images."""
    try:
        # Determine output path first
        if prompt_name == "large":
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
        
        if not story_images:
            print(f"Warning: No story images found for story_id {story_id}")
            return False
        
        # Generate instruction text
        instruction_text = get_instruction_text(
            prompt_dir,
            base_template_name=prompt_name,
            story_images=story_images,
            character_images=character_images if character_images else None,
            character_names=character_names
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

def run(story_ids, output_dir, prompt_dir, prompt_name, seed, vwp_df):
    """Process all stories and save outputs."""
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for story_id in tqdm(story_ids, total=len(story_ids)):
        if process_story(story_id, output_dir, prompt_dir, prompt_name, seed, vwp_df):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete: {successful} successful, {failed} failed")


def main(args):
    # Initialize client with the provided server URL
    initialize_client(args.server_url)
    print(f"Initialized client with server: http://{args.server_url}/v1")
    
    # Extract story IDs from CSV and load the dataframe
    _, story_ids = extract_story_ids_from_csv(args.csv_file)
    print(f"Found {len(story_ids)} unique stories in CSV")
    
    # Load character names from VWP dataset
    vwp_df = load_character_names_dataframe()
    print(f"Loaded {len(vwp_df)} rows from VWP dataset")
    
    template_dir = args.template_dir
    template_name = args.template_name
    
    output_dir = os.path.join(args.output_dir, f'prompt-{template_name}-outputs')
    os.makedirs(output_dir, exist_ok=True)

    if template_name in ['original', 'medium']:
        print(f"Running with template: {template_name}, seed 42")
        run(story_ids, output_dir, template_dir, template_name, seed=42, vwp_df=vwp_df)
    elif template_name == 'large':
        print(f"Running with template: {template_name}, all 60 stories with seeds 42, 43, 44")
        for seed in [42, 43, 44]:
            print(f"\n=== Processing with seed {seed} ===")
            run(story_ids, output_dir, template_dir, template_name, seed=seed, vwp_df=vwp_df)
    else:
        raise ValueError(f"Unknown template name: {template_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stories using Qwen3 VL model")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file containing story_ids")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save outputs")
    parser.add_argument("--template_dir", type=str, default="/mimer/NOBACKUP/groups/naiss2025-22-1187/coherence-driven-humans/data/prompts", help="Directory containing prompt templates")
    parser.add_argument("--template_name", type=str, required=True, choices=["original", "medium", "large"], help="Template name: original, medium, or large")
    parser.add_argument("--server_url", type=str, required=True, help="Server URL in format 'hostname:port' (e.g., '10.21.30.119:47246')")

    args = parser.parse_args()
    main(args)


