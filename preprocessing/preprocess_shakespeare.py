from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from datasets import DatasetDict, Dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle 
import argparse
import pandas as pd
from collections import Counter
from typing import Dict

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process the Shakespeare dataset.")
    
    parser.add_argument('--min_size', type=int, default=600, help="Minimum number of samples.")
    parser.add_argument('--split_ratio', type=float, default=0.05, help="Ratio for splitting the dataset into train and test sets.")
    parser.add_argument('--out_file', type=str, default='./shakespeare_processed.pkl', help="Output file path for the processed dataset.")

    return parser.parse_args()


def preprocess_and_train_test_split(dataset: Dataset, word_to_indices, letter_to_vec, split_ratio=0.2, seed=14) -> DatasetDict:
    """Preprocesses the dataset and splits it into training and testing sets."""
    sentence_to_indices = []
    label_to_vecs = []
    character_ids = []

    for example in tqdm(dataset, desc="Preprocessing Data"):
        sentence = example['x']
        label = example['y']

        # Convert sentence to indices and label to vectors
        sentence_indices = np.array([word_to_indices.get(word, word_to_indices["<UNK>"]) for word in sentence], dtype=np.int64)
        sentence_to_indices.append(sentence_indices)

        label_vec = letter_to_vec[label]
        label_to_vecs.append(label_vec)

        # Collect character IDs
        character_ids.append(example['character_id'])

    # Create Dataset and split
    preprocessed_data = {
        'character_id': character_ids,
        'x': sentence_to_indices,
        'y': label_to_vecs
    }
    full_dataset = Dataset.from_dict(preprocessed_data)
    train_test_split = full_dataset.train_test_split(test_size=split_ratio, seed=seed)

    return DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })


def build_word_to_indices(dataset: Dataset) -> dict:
    """Builds a word-to-index mapping from the dataset."""
    vocab = defaultdict(lambda: len(vocab))
    vocab["<UNK>"] = 0
    
    for example in tqdm(dataset, desc="Building Vocabulary"):
        for word in example["x"]:
            _ = vocab[word]

    return dict(vocab)


def build_letter_to_vec(dataset: Dataset) -> dict:
    """Builds a label-to-vector (index) mapping."""
    unique_labels = set()

    # Add tqdm to show progress while collecting unique labels
    for example in tqdm(dataset, desc="Building Letter to Vec"):
        unique_labels.add(example["y"])
    
    # Create the label to index mapping
    return {label: idx for idx, label in enumerate(sorted(unique_labels))}


def filter_partitions_by_size(fds: FederatedDataset, min_size: int) -> Dataset:
    """Filters dataset partitions based on minimum sample size."""
    partitioner = fds.partitioners['train']
    total_partitions = partitioner.num_partitions
    filtered_partitions = []

    for part_id in tqdm(range(total_partitions), desc='Filtering Partitions'):
        partition = fds.load_partition(part_id)
        if partition.num_rows >= min_size:
            filtered_partitions.append(partition)
    
    return concatenate_datasets(filtered_partitions)

def print_stats(stats):
    """Pretty print the statistics"""
    print("\n=== Shakespeare Dataset Statistics ===")
    print(f"\nOverall Statistics:")
    print(f"Total Unique Characters/partitions: {stats['total_unique_characters']:,}")
    print(f"Total Samples: {stats['total_samples']:,}")
    print(f"\nSample Distribution:")
    print(f"Minimum Samples: {stats['min_samples']:,}")
    print(f"Maximum Samples: {stats['max_samples']:,}")
    print(f"Mean Samples: {stats['mean_samples']:.2f}")
    print(f"Median Samples: {stats['median_samples']:.2f}")
    print(f"Standard Deviation: {stats['std_dev']:.2f}")
    print(f"\nDistribution Metrics:")
    print(f"Skewness: {stats['skewness']:.2f}")
    print(f"Kurtosis: {stats['kurtosis']:.2f}")
    print(f"\nTrain/Test Split:")
    print(f"Train Unique Characters: {stats['train_unique_characters']:,}")
    print(f"Test Unique Characters: {stats['test_unique_characters']:,}")
    print(f"Train Mean Samples: {stats['train_mean_samples']:.2f}")
    print(f"Test Mean Samples: {stats['test_mean_samples']:.2f}")

def get_stats(dataset):
    # convert loaded_data into flwr_dataset object 
    # Process train dataset
    train_chars = dataset['train']['character_id']
    train_counts = Counter(train_chars)

    # Process test dataset
    test_chars = dataset['test']['character_id']
    test_counts = Counter(test_chars)

    # Combine counts
    all_counts = Counter()
    all_counts.update(train_counts)
    all_counts.update(test_counts)

    # Convert to pandas series for easy statistical analysis
    counts_series = pd.Series(all_counts)

    # Calculate statistics
    stats = {
        'min_samples': counts_series.min(),
        'max_samples': counts_series.max(),
        'mean_samples': counts_series.mean(),
        'median_samples': counts_series.median(),
        'std_dev': counts_series.std(),
        'total_unique_characters': len(counts_series),
        'total_samples': counts_series.sum(),
        # Distribution metrics
        'skewness': counts_series.skew(),
        'kurtosis': counts_series.kurtosis()
    }

    # Calculate train/test split statistics
    train_stats = {
        'train_unique_characters': len(train_counts),
        'train_total_samples': sum(train_counts.values()),
        'train_mean_samples': np.mean(list(train_counts.values()))
    }

    test_stats = {
        'test_unique_characters': len(test_counts),
        'test_total_samples': sum(test_counts.values()),
        'test_mean_samples': np.mean(list(test_counts.values()))
    }

    # Add train/test stats to main stats dictionary
    stats.update(train_stats)
    stats.update(test_stats)

    return stats

def main():
    args = parse_args()

    print("Step 1 => Loading dataset from flwr_datasets")
    partitioner = NaturalIdPartitioner(partition_by="character_id")
    fds = FederatedDataset(dataset="flwrlabs/shakespeare", partitioners={"train": partitioner})
    fds.load_partition(0)

    print(f"Step 2 => Filtering partitions with samples >= {args.min_size}")
    filtered_dataset = filter_partitions_by_size(fds, args.min_size)

    

    print("Step 3 => Building mappings")
    word_to_indices = build_word_to_indices(filtered_dataset)
    letter_to_vec = build_letter_to_vec(filtered_dataset)

    print("Step 4 => Preprocessing and splitting dataset")
    processed_dataset = preprocess_and_train_test_split(
        dataset=filtered_dataset,
        word_to_indices=word_to_indices,
        letter_to_vec=letter_to_vec,
        split_ratio=args.split_ratio
    )
    del word_to_indices
    del letter_to_vec
    del fds
    del filtered_dataset
    


    print(f"Step 5 => Saving processed dataset to {args.out_file}")
    with open(args.out_file, 'wb') as f:
        pickle.dump(processed_dataset, f)

    print(f"Processing complete. Processed file saved at: {args.out_file}")

    print(f"+++++++++++ Generating Stats +++++++++++")
    print_stats(get_stats(processed_dataset))
    


if __name__ == "__main__":
    main()
