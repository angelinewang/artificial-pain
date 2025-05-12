# 1. Load Dataset
# 2. Split data
# 3. Tokenisation

# Example terminal command to run script: 
# python intensity-dataloader.py --model_name mental/mental-roberta-base --dataset_name ESConv

# Replace pip install with subprocess
import subprocess
try:
    subprocess.check_call(["pip", "install", "transformers", "datasets", "torch", "scikit-learn", "pandas", "openpyxl", "argparse", "huggingface_hub"])
except:
    print("Package installation failed or packages already installed")

import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Replace Hugging Face login with environment variable
import os
from huggingface_hub import login

# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN environment variable not set. Some operations may fail.")

import json
import numpy as np
import requests
from io import BytesIO
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Train emotion classification model')

parser.add_argument('--model_name', type=str, default="mental/mental-roberta-base", 
                    choices=["mental/mental-roberta-base", "mental/mental-bert-base-uncased", "answerdotai/ModernBERT-base"],
                    help='Model name or path (choose from available pre-trained models)')
parser.add_argument('--dataset_name', type=str, default="ESConv", 
                    choices=["ESConv"],
                    help='Dataset name (choose from available datasets)')
parser.add_argument('--balanced', action='store_true',
                    help='Whether to balance the dataset to have equal samples per label')
parser.add_argument('--samples_per_label', type=int, default=1500,
                    help='Number of samples per label when balancing (default: 1500)')

args = parser.parse_args()

if args.model_name == "mental/mental-roberta-base":
    model_abbreviation = "mental-roberta-base"
elif args.model_name == "mental/mental-bert-base-uncased":
    model_abbreviation = "mental-bert-base-uncased"
elif args.model_name == "answerdotai/ModernBERT-base":
    model_abbreviation = "ModernBERT-base"
elif args.model_name == "AIMH/mental-bert-large-uncased":
    model_abbreviation = "mental-bert-large-uncased"
elif args.model_name == "AIMH/mental-roberta-large":
    model_abbreviation = "mental-roberta-large"
elif args.model_name == "answerdotai/ModernBERT-large":
    model_abbreviation = "ModernBERT-large"
else:
    raise ValueError(f"Invalid model name: {args.model_name}")

# Define the balancing functions for ESConv dataset
def balance_esconv_dataset(df, label_classes, target_samples_per_label=1500):
    """
    Balance the ESConv dataset to have equal samples per label.
    For labels with fewer samples than target, generate new samples by selecting
    random subsets of seeker utterances from the original dialogues.
    """
    print("\nBalancing the training set...")
    
    # Create a new balanced dataframe
    balanced_samples = []
    
    # Group by label
    grouped = df.groupby('label')
    
    for label, group in grouped:
        # Get the label name for display
        label_name = label_classes[label] if isinstance(label, (int, np.integer)) else label
        original_count = len(group)
        print(f"  Label '{label_name}': {original_count} original samples")
        
        # If we have fewer samples than target, we'll need to augment
        if original_count < target_samples_per_label:
            # How many additional samples we need
            samples_to_generate = target_samples_per_label - original_count
            print(f"    Generating {samples_to_generate} additional samples")
            
            # Add all original samples
            balanced_samples.extend(group.to_dict('records'))
            
            # Generate additional samples by selecting random subsets of utterances
            for _ in range(samples_to_generate):
                # Randomly select a dialogue from this class
                sample = group.sample(1).iloc[0].copy()
                
                # Check if we have utterances stored
                if 'utterances' in sample and isinstance(sample['utterances'], list) and len(sample['utterances']) > 1:
                    # Always include first and last utterance, randomly select from the middle
                    utterances = sample['utterances']
                    if len(utterances) > 2:
                        # Select first and last utterances
                        selected_indices = [0, len(utterances) - 1]
                        
                        # Randomly select additional utterances from the middle (if any)
                        middle_indices = list(range(1, len(utterances) - 1))
                        if middle_indices:
                            num_middle_to_select = random.randint(0, len(middle_indices))
                            if num_middle_to_select > 0:
                                selected_middle = random.sample(middle_indices, num_middle_to_select)
                                selected_indices.extend(selected_middle)
                                selected_indices.sort()  # Keep utterances in chronological order
                    else:
                        # If only 1 or 2 utterances, use all of them
                        selected_indices = list(range(len(utterances)))
                    
                    # Create new text by joining selected utterances
                    selected_utterances = [utterances[i] for i in selected_indices]
                    sample['text'] = " ".join(selected_utterances)
                    
                    # Add to balanced samples
                    balanced_samples.append(sample)
                else:
                    # If no utterances available, just duplicate the original sample
                    balanced_samples.append(sample)
        else:
            # If we have more than enough samples, randomly select the target number
            selected_samples = group.sample(target_samples_per_label, random_state=42)
            balanced_samples.extend(selected_samples.to_dict('records'))
    
    # Convert to DataFrame
    balanced_df = pd.DataFrame(balanced_samples)
    
    # Print statistics about the balanced dataset
    print("\nBalanced dataset statistics:")
    for label, count in balanced_df['label'].value_counts().items():
        label_name = label_classes[label] if isinstance(label, (int, np.integer)) else label
        print(f"  Label '{label_name}': {count} samples")
    
    return balanced_df

# 1. Dataset Loading and Preparation --------------------------------
def load_dataset():
    if args.dataset_name == "ESConv":
        url = "https://github.com/thu-coai/Emotional-Support-Conversation/raw/main/ESConv.json"
        response = requests.get(url)

        # Load JSON directly without using zipfile
        data = json.loads(response.content)

        # Create a list to store all samples
        samples = []

        # Process each dialogue
        for dialogue_idx, dialogue in enumerate(data):
            if 'emotion_type' in dialogue and dialogue['emotion_type'] is not None:
                label = dialogue['emotion_type']

                # Extract all seeker utterances and store them separately
                seeker_utterances = []
                for utterance in dialogue['dialog']:
                    if utterance['speaker'] == 'seeker' and 'content' in utterance:
                        seeker_utterances.append(utterance['content'].strip())

                # Skip if no seeker utterances found
                if not seeker_utterances:
                    continue

                # Concatenate all seeker utterances into one string
                concatenated_content = " ".join(seeker_utterances)

                # Extract initial and final emotion intensity if available
                initial_emotion_intensity = None
                final_emotion_intensity = None

                if 'survey_score' in dialogue and 'seeker' in dialogue['survey_score']:
                    seeker_survey = dialogue['survey_score']['seeker']
                    if 'initial_emotion_intensity' in seeker_survey:
                        # Convert to float if possible
                        try:
                            initial_emotion_intensity = float(seeker_survey['initial_emotion_intensity'])
                        except (ValueError, TypeError):
                            initial_emotion_intensity = None

                    if 'final_emotion_intensity' in seeker_survey:
                        # Convert to float if possible
                        try:
                            final_emotion_intensity = float(seeker_survey['final_emotion_intensity'])
                        except (ValueError, TypeError):
                            final_emotion_intensity = None

                # Add the sample with all required information
                samples.append({
                    'text': concatenated_content,
                    'label': label,
                    'dialogue_idx': dialogue_idx,
                    'initial_emotion_intensity': initial_emotion_intensity,
                    'final_emotion_intensity': final_emotion_intensity,
                    'problem_type': dialogue.get('problem_type', None),
                    'experience_type': dialogue.get('experience_type', None),
                    'situation': dialogue.get('situation', None),
                    'utterances': seeker_utterances  # Store the original utterances
                })

        # Convert to DataFrame
        df = pd.DataFrame(samples)
        df = df.dropna(subset=['text', 'label'])  # Only drop rows with missing text or label

        # Print unique labels before encoding
        unique_labels = df['label'].unique()
        print(f"Unique labels before encoding: {unique_labels}")

        print(f"Number of unique labels: {len(unique_labels)}")
        print(f"Total number of dialogues with seeker utterances: {len(df)}")

        # Count samples per label
        label_counts = df['label'].value_counts()
        print("\nSamples per label:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

        # Filter out labels with fewer than 15 samples for ESConv
        min_samples_threshold = 15
        rare_labels = label_counts[label_counts < min_samples_threshold].index.tolist()
        if rare_labels:
            print(f"\nRemoving {len(rare_labels)} labels with fewer than {min_samples_threshold} samples: {rare_labels}")
            df = df[~df['label'].isin(rare_labels)]
            print(f"Remaining samples: {len(df)}")

        # Convert string labels to numerical indices
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])

        # Print the classes for reference
        print(f"Classes: {le.classes_}")
        print(f"Number of classes: {len(le.classes_)}")
        print(f"Label range: {df['label'].min()} to {df['label'].max()}")

        # Make sure your model's config matches the number of classes
        num_labels = len(le.classes_)
        print(f"Make sure your model is configured for {num_labels} classes")

        # Print emotion intensity statistics
        print("\nEmotion Intensity Statistics:")
        print(f"Initial emotion intensity available: {df['initial_emotion_intensity'].notna().sum()} dialogues")
        print(f"Final emotion intensity available: {df['final_emotion_intensity'].notna().sum()} dialogues")

        # Safely calculate min/max for emotion intensities
        if df['initial_emotion_intensity'].notna().sum() > 0:
            try:
                min_initial = df['initial_emotion_intensity'].min()
                max_initial = df['initial_emotion_intensity'].max()
                print(f"Initial emotion intensity range: {min_initial} to {max_initial}")
            except TypeError:
                print("Warning: Mixed types in initial_emotion_intensity")

        if df['final_emotion_intensity'].notna().sum() > 0:
            try:
                min_final = df['final_emotion_intensity'].min()
                max_final = df['final_emotion_intensity'].max()
                print(f"Final emotion intensity range: {min_final} to {max_final}")
            except TypeError:
                print("Warning: Mixed types in final_emotion_intensity")

        # Split the dataset first to ensure test and validation sets remain unbalanced
        try:
            train_df, temp_df = train_test_split(
                df, test_size=0.3, random_state=42, stratify=df['label']
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.6667, random_state=42, stratify=temp_df['label']
            )
        except ValueError as e:
            print(f"Warning: {e}")
            print("Falling back to non-stratified split")
            train_df, temp_df = train_test_split(
                df, test_size=0.3, random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.6667, random_state=42
            )
            
        # Balance ONLY the training dataset if requested
        if args.balanced:
            print(f"Balancing ONLY the training set of ESConv dataset to have {args.samples_per_label} samples per label...")
            train_df = balance_esconv_dataset(train_df, le.classes_, target_samples_per_label=args.samples_per_label)
            print(f"Training set balanced. Total training samples: {len(train_df)}")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Save the label encoder classes for later use
    label_mapping = {i: label for i, label in enumerate(le.classes_)}

    return DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
        'validation': Dataset.from_pandas(val_df.reset_index(drop=True)),
        'test': Dataset.from_pandas(test_df.reset_index(drop=True))
    }), num_labels, label_mapping

dataset, num_labels, label_mapping = load_dataset()

# Save dataset statistics
def save_dataset_statistics(dataset_dict, save_path):
    stats = {}

    # Overall statistics
    total_samples = sum(len(split) for split in dataset_dict.values())
    stats["total_samples"] = total_samples

    # Per-split statistics
    for split_name, split_dataset in dataset_dict.items():
        split_stats = {
            "samples": len(split_dataset),
            "percentage": f"{len(split_dataset) / total_samples * 100:.2f}%"
        }

        # Label distribution
        if 'label' in split_dataset.features:
            label_counts = {}
            for label in split_dataset['label']:
                label_str = label_mapping.get(label, str(label))
                label_counts[label_str] = label_counts.get(label_str, 0) + 1
            split_stats["label_distribution"] = label_counts

        if args.dataset_name == "ESConv":
            # Emotion intensity statistics if available
            for intensity_type in ['initial_emotion_intensity', 'final_emotion_intensity']:
                if intensity_type in split_dataset.features:
                    # Filter out None values
                    intensity_values = [v for v in split_dataset[intensity_type] if v is not None]
                    if intensity_values:
                        split_stats[f"{intensity_type}_stats"] = {
                            "available": len(intensity_values),
                            "min": min(intensity_values),
                            "max": max(intensity_values),
                            "mean": sum(intensity_values) / len(intensity_values)
                        }

        stats[split_name] = split_stats

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Dataset statistics saved to {save_path}")

# Save dataset splits to Disk
def save_dataset_splits(dataset_dict, save_dir):
    import os
    import pandas as pd

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save each split as a CSV file
    for split_name, split_dataset in dataset_dict.items():
        # Convert to pandas DataFrame
        df = pd.DataFrame(split_dataset)

        # Save to CSV
        save_path = f"{save_dir}/{split_name}_split.csv"
        df.to_csv(save_path, index=False)
        print(f"Saved {split_name} split to {save_path}")

    # Save dataset info
    split_sizes = {split: len(dataset) for split, dataset in dataset_dict.items()}

    if not args.balanced:
        with open(f"{save_dir}/dataset_info.json", "w") as f:
            json.dump({
                "split_sizes": split_sizes,
                "total_examples": sum(split_sizes.values()),
                "features": list(dataset_dict["train"].features.keys())
            }, f, indent=4)
    else:
        with open(f"{save_dir}/balanced_dataset_info.json", "w") as f:
            json.dump({
                "split_sizes": split_sizes,
                "total_examples": sum(split_sizes.values()),
                "features": list(dataset_dict["train"].features.keys())
            }, f, indent=4)

    if not args.balanced:
        print(f"Dataset info saved to /{save_dir}/dataset_info.json")
    else:
        print(f"Dataset info saved to /{save_dir}/balanced_dataset_info.json")

# Define the save directory in disk
dataset_save_dir = f"/{args.dataset_name}/{'balanced_' if args.balanced else ''}dataset_splits"

# Create directory if it doesn't exist
import os
os.makedirs(dataset_save_dir, exist_ok=True)

# Only save label mapping, dataset statistics and dataset splits once
# Save the label mapping for later reference
label_mapping_path = f"/{args.dataset_name}/{'balanced_' if args.balanced else ''}label_mapping.json"
# Need to ensure the parent directory exists
os.makedirs(os.path.dirname(label_mapping_path), exist_ok=True)
if not os.path.exists(label_mapping_path):
    with open(label_mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=4)
    print(f"Label mapping saved to {label_mapping_path}")
else:
    print(f"Label mapping already exists at {label_mapping_path}")

# Save dataset statistics
stats_path = f"/{args.dataset_name}/{'balanced_' if args.balanced else ''}dataset_statistics.json"
# Parent directory is already created above, but adding for clarity
os.makedirs(os.path.dirname(stats_path), exist_ok=True)
if not os.path.exists(stats_path):
    save_dataset_statistics(dataset, stats_path)
else:
    print(f"Dataset statistics already exist at {stats_path}")

# Save the original dataset splits
if not any(os.path.exists(f"/{dataset_save_dir}/{split}_split.csv") for split in ["train", "validation", "test"]):
    save_dataset_splits(dataset, dataset_save_dir)
else:
    print(f"Dataset splits already exist in /{dataset_save_dir}")


# 2. Tokenization --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Ensure emotion intensity columns are preserved in the tokenized dataset
# Set format with emotion intensity columns included
columns_to_keep = ["input_ids", "attention_mask", "labels"]

# Add emotion intensity columns if they exist in the dataset
if "initial_emotion_intensity" in tokenized_datasets["train"].features:
    columns_to_keep.append("initial_emotion_intensity")
if "final_emotion_intensity" in tokenized_datasets["train"].features:
    columns_to_keep.append("final_emotion_intensity")

tokenized_datasets.set_format("torch", columns=columns_to_keep)

# Save the tokenized dataset splits
tokenized_save_dir = f"/{args.dataset_name}/{model_abbreviation}/{'balanced_' if args.balanced else ''}tokenized_splits"
save_dataset_splits(tokenized_datasets, tokenized_save_dir)

# Save the indices of examples in each split
def save_split_indices(dataset_dict, save_dir):
    import os
    import json

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Extract indices for each split
    split_indices = {}
    for split_name, split_dataset in dataset_dict.items():
        # Dataset objects don't have an 'indices' attribute
        # Instead, just save the range of indices (0 to length-1)
        split_indices[split_name] = list(range(len(split_dataset)))
        
    # Save the indices
    if not args.balanced:
        with open(f"{save_dir}/split_indices.json", "w") as f:
            json.dump(split_indices, f, indent=4)
    else:
        with open(f"{save_dir}/balanced_split_indices.json", "w") as f:
            json.dump(split_indices, f, indent=4)

    if not args.balanced:
        print(f"Split indices saved to {save_dir}/split_indices.json")
    else:
        print(f"Split indices saved to {save_dir}/balanced_split_indices.json")

# Save the indices of examples in each split
indices_save_dir = f"/{args.dataset_name}/{model_abbreviation}/{'balanced_' if args.balanced else ''}tokenized_splits"
save_split_indices(tokenized_datasets, indices_save_dir)
