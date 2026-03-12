import json
import os
import argparse
from typing import List, Dict, Tuple
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Paths
DATA_DIR = '/mimer/NOBACKUP/groups/naiss2025-22-1187/data/bertopic_inputs'
DEFAULT_CACHE_DIR = '/mimer/NOBACKUP/groups/naiss2024-6-297/cache/bertopic_bootstrapped'

def load_full_dataset() -> Tuple[List[str], pd.DataFrame]:
    """Load the full dataset (all texts including all human-large seeds)."""
    stories_path = os.path.join(DATA_DIR, 'all_stories_texts.json')
    metadata_path = os.path.join(DATA_DIR, 'all_stories_metadata.csv')
    
    with open(stories_path, 'r') as f:
        stories = json.load(f)
    
    metadata = pd.read_csv(metadata_path)
    
    return stories, metadata

def process_sentences(stories: List[str]) -> Tuple[List[str], Dict, Dict]:
    """Process stories into sentences and create mapping dictionaries."""
    doc_sentence_mapping = {}
    all_sentences = []
    sentence_to_doc = {}

    for doc_id, doc in enumerate(stories):
        sentences = [sent.strip() for sent in doc.split('[SENT]') if sent.strip()]
        
        start_idx = len(all_sentences)
        doc_sentence_mapping[doc_id] = list(range(start_idx, start_idx + len(sentences)))
        
        all_sentences.extend(sentences)
        
        for sent_idx in range(start_idx, start_idx + len(sentences)):
            sentence_to_doc[sent_idx] = doc_id

    return all_sentences, doc_sentence_mapping, sentence_to_doc

def get_embeddings(sentences: List[str], 
                   embedding_model: SentenceTransformer,
                   cache_file: str,
                   batch_size: int = 64) -> np.ndarray:
    """Get or compute embeddings with caching."""
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        return np.load(cache_file)
    
    print("Computing embeddings...")
    import torch
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    embedding_model.to('cuda')
    if hasattr(embedding_model, 'model'):
        model_base = embedding_model.model[0].auto_model
        model_base.gradient_checkpointing_enable()
        model_base.config.use_cache = False

    embeddings_list = []
    chunk_size = 1000
    
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i + chunk_size]
        chunk_embeddings = embedding_model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=True,
            device='cuda',
            num_workers=0
        )
        embeddings_list.append(chunk_embeddings)
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    embeddings = np.concatenate(embeddings_list)
    
    print(f"Caching embeddings to {cache_file}...")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, embeddings)
    
    return embeddings

def main():
    parser = argparse.ArgumentParser(description='Test BERTopic on full dataset')
    parser.add_argument('--bootstrap-idx', type=int, required=True,
                        help='Bootstrap index of trained model (0-9)')
    parser.add_argument('--nr-topics', type=int, required=True,
                        help='Number of topics in trained model')
    parser.add_argument('--cache-dir', type=str, default=DEFAULT_CACHE_DIR,
                        help='Root cache directory (should match training --cache-dir)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (overrides default)')
    args = parser.parse_args()
    
    bootstrap_idx = args.bootstrap_idx
    nr_topics = args.nr_topics
    cache_dir = args.cache_dir
    
    # Path to trained model
    model_path = os.path.join(
        cache_dir,
        "models",
        f"bootstrap_{bootstrap_idx:02d}",
        f"topics_{nr_topics}"
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Output directory
    if args.output_dir is None:
        output_dir = os.path.join(
            cache_dir,
            "inference",
            f"bootstrap_{bootstrap_idx:02d}",
            f"topics_{nr_topics}"
        )
    else:
        output_dir = args.output_dir
    
    # Skip if already done
    if os.path.exists(os.path.join(output_dir, 'full_results.csv')):
        print(f"SKIPPING: Results already exist in {output_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Testing BERTopic Model")
    print(f"  Bootstrap: {bootstrap_idx}")
    print(f"  Nr Topics: {nr_topics}")
    print(f"  Model Path: {model_path}")
    print(f"  Output Dir: {output_dir}")
    print(f"{'='*80}")
    
    # Load full dataset
    print("\nStep 1: Loading Full Dataset")
    stories, metadata = load_full_dataset()
    print(f"Loaded {len(stories)} stories")
    print(f"Metadata columns: {list(metadata.columns)}")
    
    # Process sentences
    all_sentences, doc_sentence_mapping, sentence_to_doc = process_sentences(stories)
    print(f"Total sentences: {len(all_sentences)}")
    
    # Load embeddings for full dataset
    print("\nStep 2: Loading/Computing Embeddings for Full Dataset")
    embedding_model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral", device='cuda')
    
    # Store embeddings in the cache folder
    embeddings_dir = os.path.join(cache_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    cache_file = os.path.join(embeddings_dir, 'embeddings_full_dataset.npy')
    embeddings = get_embeddings(all_sentences, embedding_model, cache_file)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Load trained model
    print("\nStep 3: Loading Trained BERTopic Model")
    topic_model = BERTopic.load(model_path)
    print(f"Model loaded with {len(topic_model.get_topics())} topics")
    
    # Transform (predict topics for full dataset)
    print("\nStep 4: Predicting Topics for Full Dataset")
    topics, probs = topic_model.transform(all_sentences, embeddings=embeddings)
    
    # Aggregate sentence-level topics to document level
    print("\nStep 5: Aggregating Results to Document Level")
    doc_results = []
    
    for doc_id, sent_indices in tqdm(doc_sentence_mapping.items(), desc="Aggregating"):
        doc_topics = [topics[i] for i in sent_indices]
        doc_probs = [probs[i] for i in sent_indices]
        
        # Get most common topic (mode)
        from collections import Counter
        topic_counts = Counter(doc_topics)
        main_topic = topic_counts.most_common(1)[0][0]
        
        # Average probabilities
        if isinstance(doc_probs[0], np.ndarray):
            avg_probs = np.mean(doc_probs, axis=0)
        else:
            avg_probs = np.mean(doc_probs)
        
        doc_results.append({
            'doc_id': doc_id,
            'main_topic': main_topic,
            'topic_counts': dict(topic_counts),
            'n_sentences': len(sent_indices),
            'avg_prob': float(np.max(avg_probs)) if isinstance(avg_probs, np.ndarray) else float(avg_probs)
        })
    
    # Merge with metadata
    results_df = pd.DataFrame(doc_results)
    results_df = results_df.merge(metadata, left_on='doc_id', right_on='story_index', how='left')
    
    # Save results
    print("\nStep 6: Saving Results")
    
    # Save full results
    results_df.to_csv(os.path.join(output_dir, 'full_results.csv'), index=False)
    
    # Save sentence-level topics
    np.save(os.path.join(output_dir, 'sentence_topics.npy'), np.array(topics))
    np.save(os.path.join(output_dir, 'sentence_probs.npy'), np.array(probs))
    
    # Save mapping info
    mapping_info = {
        'bootstrap_idx': bootstrap_idx,
        'nr_topics': nr_topics,
        'n_stories': len(stories),
        'n_sentences': len(all_sentences),
        'model_path': model_path
    }
    with open(os.path.join(output_dir, 'inference_info.json'), 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total documents: {len(results_df)}")
    print(f"Unique topics assigned: {results_df['main_topic'].nunique()}")
    print(f"\nDocuments per model_type:")
    print(results_df['model_type'].value_counts())
    print(f"\nDocuments per (model_type, prompt_type):")
    print(results_df.groupby(['model_type', 'prompt_type']).size())
    
    print(f"\n✓ Results saved to {output_dir}")

if __name__ == "__main__":
    main()
