import json
import locale
import os
from typing import List, Dict, Tuple, Optional
from tqdm.auto import tqdm
import argparse

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN

import numpy.lib.format as nplf

# Set UTF-8 encoding
locale.getpreferredencoding = lambda: "UTF-8"

# Paths to data
DATA_DIR = '../data/bertopic_inputs'
BALANCED_TRAIN_DIR = os.path.join(DATA_DIR, 'balanced_train_sets')
DEFAULT_CACHE_DIR = '/mimer/NOBACKUP/groups/naiss2024-6-297/cache/bertopic_bootstrapped'

def load_balanced_training_set(bootstrap_idx: int) -> List[str]:
    """Load a specific bootstrapped balanced training set."""
    texts_path = os.path.join(BALANCED_TRAIN_DIR, f'train_texts_bootstrap_{bootstrap_idx:02d}.json')
    with open(texts_path, 'r') as f:
        stories = json.load(f)
    return stories

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

def create_topic_model(
    nr_topics: Optional[int] = 500,
    embedding_model: SentenceTransformer = None,
    min_topic_size: Optional[int] = None,
) -> BERTopic:
    """Create and configure BERTopic model.

    Explicitly controlled parameters:
        - min_topic_size / min_cluster_size  (HDBSCAN + BERTopic)
        - nr_topics                          (BERTopic)
    """
    if embedding_model is None:
        embedding_model = SentenceTransformer(
            "Salesforce/SFR-Embedding-Mistral",
            device='cuda'
        )

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        output_type='numpy',
    )

    hdbscan_kwargs = {
        'metric': 'euclidean',
        'cluster_selection_method': 'eom',
        'prediction_data': True,
        'output_type': 'numpy',
    }
    if min_topic_size is not None:
        hdbscan_kwargs['min_cluster_size'] = min_topic_size
    hdbscan_model = HDBSCAN(**hdbscan_kwargs)


    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        verbose=True,
    )


def run_pilot_discovery(
    bootstrap_idx: int,
    all_sentences: List[str],
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    stories: List[str],
    doc_sentence_mapping: Dict,
    output_dir: str,
    min_topic_size: Optional[int] = None,
):
    print(f"Bootstrap {bootstrap_idx}: Pilot discovery run (nr_topics=None)")

    topic_model = create_topic_model(
        nr_topics=None,
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
    )
    topics, _ = topic_model.fit_transform(all_sentences, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()
    discovered_topics_excl_outlier = int((topic_info['Topic'] != -1).sum())
    discovered_topics_incl_outlier = int(topic_info.shape[0])
    outlier_count = int((np.array(topics) == -1).sum())

    os.makedirs(output_dir, exist_ok=True)

    topic_info.to_csv(
        os.path.join(output_dir, f'topic_info_bootstrap_{bootstrap_idx:02d}.csv'),
        index=False,
    )

    summary = {
        'bootstrap_idx': bootstrap_idx,
        'n_stories': len(stories),
        'n_sentences': len(all_sentences),
        'discovered_topics_excl_outlier': discovered_topics_excl_outlier,
        'discovered_topics_incl_outlier': discovered_topics_incl_outlier,
        'outlier_sentence_count': outlier_count,
    }

    with open(
        os.path.join(output_dir, f'discovery_summary_bootstrap_{bootstrap_idx:02d}.json'),
        'w',
    ) as f:
        json.dump(summary, f, indent=2)

    mapping_info = {
        'bootstrap_idx': bootstrap_idx,
        'n_stories': len(stories),
        'n_sentences': len(all_sentences),
        'doc_sentence_mapping': {str(k): v for k, v in doc_sentence_mapping.items()},
    }
    with open(os.path.join(output_dir, f'mapping_info_bootstrap_{bootstrap_idx:02d}.json'), 'w') as f:
        json.dump(mapping_info, f, indent=2)

    print('\nPilot discovery summary:')
    print(summary)

def get_embeddings(sentences: List[str], 
                  embedding_model: SentenceTransformer,
                  cache_file: str,
                  batch_size: int = None) -> np.ndarray:
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        return np.load(cache_file)
    
    print("Computing embeddings...")
    
    import torch
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if batch_size is None:
        batch_size = 64
    
    print(f"Using batch size: {batch_size}")

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

def get_output_dir(bootstrap_idx: int, nr_topics: int, base_dir: str) -> str:
    """Build output directory with bootstrap index and number of topics."""
    output_dir = os.path.join(
        base_dir,
        "models",
        f"bootstrap_{bootstrap_idx:02d}",
        f"topics_{nr_topics}"
    )
    
    return output_dir

def safe_save_large_array(filepath: str, arr):
    """Save large numpy array using memory mapping."""
    try:
        if isinstance(arr, list):
            arr = np.array(arr)
        
        shape, dtype = arr.shape, arr.dtype
        
        fp = nplf.open_memmap(
            filepath, 
            mode='w+', 
            dtype=dtype, 
            shape=shape
        )
        
        chunk_size = 1000
        for i in range(0, shape[0], chunk_size):
            end = min(i + chunk_size, shape[0])
            fp[i:end] = arr[i:end]
        
        fp.flush()
        del fp
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise Exception(f"Error saving array to {filepath}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Train BERTopic on balanced bootstrap sets')
    parser.add_argument('--bootstrap-idx', type=int, default=0,
                        help='Bootstrap index (0-9)')
    parser.add_argument('--nr-topics-start', type=int, default=80,
                        help='Starting (highest) number of topics')
    parser.add_argument('--nr-topics-end', type=int, default=5,
                        help='Ending (lowest) number of topics')
    parser.add_argument('--nr-topics-step', type=int, default=5,
                        help='Step size for reducing topics')
    parser.add_argument('--pilot-discovery', action='store_true',
                        help='Run one discovery fit with nr_topics=None and save discovered topic counts per bootstrap')
    parser.add_argument('--cache-dir', type=str, default=DEFAULT_CACHE_DIR,
                        help='Output/cache root directory for this run (embeddings + outputs)')
    parser.add_argument('--min-topic-size', type=int, default=None,
                        help='Minimum topic/cluster size. If omitted, use default model settings.')
    args = parser.parse_args()
    
    bootstrap_idx = args.bootstrap_idx
    
    print(f"Training BERTopic on Bootstrap Set {bootstrap_idx}")
    print()
    
    print("\nLoading Balanced Training Data")
    stories = load_balanced_training_set(bootstrap_idx)
    print(f"Loaded {len(stories)} stories from bootstrap set {bootstrap_idx}")
    print(f"min_topic_size: {args.min_topic_size if args.min_topic_size is not None else 'default'}")
    
    # Process sentences
    all_sentences, doc_sentence_mapping, sentence_to_doc = process_sentences(stories)
    print(f"Total number of sentences: {len(all_sentences)}")
    
    # Load or compute embeddings for this bootstrap set
    print("Step 2: Computing/Loading Embeddings")
    embedding_model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral", device='cuda')
    
    # Store embeddings in the new cache folder
    embeddings_dir = os.path.join(args.cache_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    cache_file = os.path.join(embeddings_dir, f'embeddings_bootstrap_{bootstrap_idx:02d}.npy')
    embeddings = get_embeddings(
        all_sentences,
        embedding_model,
        cache_file=cache_file
    )
    print(f"Embeddings shape: {embeddings.shape}")
    
    if args.pilot_discovery:
        print("Step 3: Running Pilot Discovery")
        run_pilot_discovery(
            bootstrap_idx=bootstrap_idx,
            all_sentences=all_sentences,
            embeddings=embeddings,
            embedding_model=embedding_model,
            stories=stories,
            doc_sentence_mapping=doc_sentence_mapping,
            output_dir=args.cache_dir,
            min_topic_size=args.min_topic_size,
        )
        print(f"Bootstrap {bootstrap_idx}: Pilot discovery completed!")
        return

    print("Step 3: Running Topic Modeling Pipeline")
    
    # Iterate through different numbers of topics
    # Start from 300, then reduce by 25: 275, 250, 225, ..., 25
    for nr_topics in range(args.nr_topics_start, args.nr_topics_end - 1, -args.nr_topics_step):

        print(f"Bootstrap {bootstrap_idx}: Processing model with {nr_topics} topics")
        print()
        
        import torch
        torch.cuda.empty_cache()
        
        output_dir = get_output_dir(bootstrap_idx, nr_topics, base_dir=args.cache_dir)
        
        if os.path.exists(os.path.join(output_dir, 'topic_info.csv')):
            print(f"SKIPPING: Results already exist in {output_dir}")
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
        
        # Create topic model with specified number of topics
        topic_model = create_topic_model(
            nr_topics=nr_topics,
            embedding_model=embedding_model,
            min_topic_size=args.min_topic_size,
        )
        
        print("\nFitting Topic Model")
        
        # Fit and transform using pre-computed embeddings
        topics, probs = topic_model.fit_transform(all_sentences, embeddings=embeddings)
        
        print("Step 4: Saving Results")
        with tqdm(total=4, desc="Saving outputs") as pbar:
            # Save document-topic assignments
            safe_save_large_array(os.path.join(output_dir, 'topics.npy'), topics)
            pbar.update(1)
            
            # Save topic probabilities
            safe_save_large_array(os.path.join(output_dir, 'probabilities.npy'), probs)
            pbar.update(1)
            
            # Save detailed topic information
            topic_info = topic_model.get_topic_info()
            topic_info.to_csv(os.path.join(output_dir, 'topic_info.csv'))
            pbar.update(1)
            
            # Save complete model
            topic_model.save(
                path=output_dir,
                serialization="safetensors",
                save_embedding_model="Salesforce/SFR-Embedding-Mistral"
            )
            pbar.update(1)
        
        # Save mapping info, used for sentence-story mapping
        mapping_info = {
            'bootstrap_idx': bootstrap_idx,
            'nr_topics': nr_topics,
            'n_stories': len(stories),
            'n_sentences': len(all_sentences),
            'doc_sentence_mapping': {str(k): v for k, v in doc_sentence_mapping.items()}
        }
        with open(os.path.join(output_dir, 'mapping_info.json'), 'w') as f:
            json.dump(mapping_info, f, indent=2)
        
        print(f"\nFound {len(topic_model.get_topics())} topics")
        print("Top topics by size:")
        print(topic_info.head())
    
    print(f"\nBootstrap {bootstrap_idx}: All topic models completed!")

if __name__ == "__main__":
    main()