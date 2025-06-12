import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, f1_score, recall_score
import argparse

def calculate_mse(predictions, targets):
    """Calculate Mean Squared Error between predictions and targets."""
    # Convert inputs to numpy arrays and handle NaN values
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Only consider pairs where both prediction and target are valid (not NaN)
    valid_mask = ~np.isnan(targets)
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    # Calculate MSE
    mse = np.mean((predictions - targets) ** 2)
    return mse, len(predictions)

def calculate_micro_mse(predictions, targets):
    """
    Calculate micro-averaged MSE.
    This pools all predictions and targets together before calculating MSE.
    """
    # Convert inputs to numpy arrays and handle NaN values
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Only consider pairs where both prediction and target are valid (not NaN)
    valid_mask = ~np.isnan(targets)
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    # Calculate squared errors for each prediction
    squared_errors = (predictions - targets) ** 2
    
    # Micro-average: sum all errors and divide by total number
    micro_mse = np.sum(squared_errors) / len(squared_errors)
    return micro_mse, len(predictions)

def calculate_means(df):
    """Calculate mean values for emotion intensities."""
    initial_mean = df['initial_emotion_intensity'].mean() if 'initial_emotion_intensity' in df.columns else None
    final_mean = df['final_emotion_intensity'].mean() if 'final_emotion_intensity' in df.columns else None
    
    if initial_mean is not None and final_mean is not None:
        changes = df['final_emotion_intensity'] - df['initial_emotion_intensity']
        change_mean = changes.mean()
        change_count = changes.notna().sum()
    else:
        change_mean = None
        change_count = 0
    
    initial_count = df['initial_emotion_intensity'].notna().sum() if 'initial_emotion_intensity' in df.columns else 0
    final_count = df['final_emotion_intensity'].notna().sum() if 'final_emotion_intensity' in df.columns else 0
    
    return {
        'initial': (initial_mean, initial_count),
        'final': (final_mean, final_count),
        'change': (change_mean, change_count)
    }

def calculate_emotion_type_stats(df):
    """Calculate emotion type distribution and majority class."""
    emotion_counts = df['label'].value_counts()
    total_samples = len(df)
    emotion_probs = emotion_counts / total_samples
    majority_class = emotion_counts.index[0]
    
    return {
        'distribution': emotion_probs,
        'majority_class': majority_class,
        'majority_prob': emotion_probs[majority_class]
    }

def calculate_weighted_metrics(y_true, y_pred):
    """
    Calculate weighted-averaged F1 and recall scores.
    Weighted averaging accounts for class imbalance by computing the average of binary metrics, 
    weighted by support (the number of true instances for each class).
    """
    # Calculate weighted F1 and recall scores
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    
    return {
        'weighted_f1': weighted_f1,
        'weighted_recall': weighted_recall,
        'support': len(y_true)
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate baseline metrics for emotion classification and intensity prediction')
    parser.add_argument('--num_tasks', type=str, choices=['single-task', 'multi-task'], default='single-task',
                      help='Whether to run single-task (classification only) or multi-task (classification and intensity)')
    parser.add_argument('--dataset', type=str, choices=['SAD', 'ESConv'], default='ESConv',
                      help='Which dataset to use')
    args = parser.parse_args()

    # Set paths based on dataset choice
    if args.dataset == 'SAD':
        base_path = "./classification/SAD/dataset_splits"
    else:  # ESConv
        base_path = "./intensity/ESConv/dataset_splits"

    # Load training data first
    train_path = f"{base_path}/train_split.csv"
    try:
        train_df = pd.read_csv(train_path)
        train_emotion_stats = calculate_emotion_type_stats(train_df)
        
        print("\nTraining Data Statistics:")
        print("\nEmotion Type Distribution in Training Data:")
        for emotion, prob in train_emotion_stats['distribution'].items():
            print(f"{emotion}: {prob:.4f}")
        print(f"Majority class: {train_emotion_stats['majority_class']} ({train_emotion_stats['majority_prob']:.4f})")

        # Only calculate intensity statistics for multi-task mode with ESConv dataset
        if args.num_tasks == 'multi-task' and args.dataset == 'ESConv':
            train_means = calculate_means(train_df)
            print(f"\nMean Initial Emotion Intensity: {train_means['initial'][0]:.4f} (from {train_means['initial'][1]} samples)")
            print(f"Mean Final Emotion Intensity: {train_means['final'][0]:.4f} (from {train_means['final'][1]} samples)")
            print(f"Mean Emotion Intensity Change: {train_means['change'][0]:.4f} (from {train_means['change'][1]} samples)")

    except FileNotFoundError:
        print(f"Error: Could not find training data at {train_path}")
        return

    # Load test data
    test_path = f"{base_path}/test_split.csv"
    try:
        df = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: Could not find test data at {test_path}")
        return

    # Calculate emotion type prediction metrics
    # Strategy 1: Always predict majority class
    majority_predictions = [train_emotion_stats['majority_class']] * len(df)
    majority_accuracy = accuracy_score(df['label'], majority_predictions)
    print(f"\nEmotion Type Classification - Majority Class Baseline:")
    print(f"Accuracy: {majority_accuracy:.4f}")
    
    # Calculate and display weighted metrics for majority class
    majority_metrics = calculate_weighted_metrics(df['label'].values, np.array(majority_predictions))
    print("\nMajority Class Weighted Metrics:")
    print(f"Weighted F1: {majority_metrics['weighted_f1']:.4f}")
    print(f"Weighted Recall: {majority_metrics['weighted_recall']:.4f}")
    print(f"Support: {majority_metrics['support']}")

    # Only calculate intensity metrics for multi-task mode with ESConv dataset
    if args.num_tasks == 'multi-task' and args.dataset == 'ESConv':
        # Use training means as constant predictions
        initial_pred = train_means['initial'][0]
        final_pred = train_means['final'][0]
        change_pred = train_means['change'][0]

        # Calculate micro-averaged MSE for initial intensity
        initial_mse, initial_count = calculate_micro_mse(
            [initial_pred] * len(df),
            df['initial_emotion_intensity'].values
        )
        print(f"\nInitial Emotion Intensity Micro-averaged MSE (using training mean {initial_pred:.4f}):")
        print(f"MSE: {initial_mse:.4f}")
        print(f"Number of samples: {initial_count}")

        # Calculate micro-averaged MSE for final intensity
        final_mse, final_count = calculate_micro_mse(
            [final_pred] * len(df),
            df['final_emotion_intensity'].values
        )
        print(f"\nFinal Emotion Intensity Micro-averaged MSE (using training mean {final_pred:.4f}):")
        print(f"MSE: {final_mse:.4f}")
        print(f"Number of samples: {final_count}")

        # Calculate micro-averaged MSE for intensity change
        # First calculate actual changes
        actual_changes = df['final_emotion_intensity'] - df['initial_emotion_intensity']
        change_mse, change_count = calculate_micro_mse(
            [change_pred] * len(df),
            actual_changes.values
        )
        print(f"\nEmotion Intensity Change Micro-averaged MSE (using training mean {change_pred:.4f}):")
        print(f"MSE: {change_mse:.4f}")
        print(f"Number of samples: {change_count}")

if __name__ == "__main__":
    main()