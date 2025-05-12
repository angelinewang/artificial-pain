# Used for all MentalRoBERTa, MentalBERT, and ModernBERT 

# Example terminal commands to run the script:

# Command to run script: 
# python intensity-model.py --model_name mental/mental-roberta-base --dataset_name ESConv --head_num 4 --model_metric accuracy

import subprocess
try:
    subprocess.check_call(["pip", "install", "transformers", "datasets", "evaluate", "torch", "scikit-learn", "pandas", "openpyxl", "wandb", "argparse", "huggingface_hub", "safetensors"])
except:
    print("Package installation failed or packages already installed")

import wandb
import evaluate
import argparse
import random
import math

import json
import os
import torch
from safetensors.torch import load_file
from torch import nn
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback, ModernBertModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr

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

# Set up Weights & Biases with environment variable
import wandb
wandb_api_key = os.environ.get("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    print("Warning: WANDB_API_KEY environment variable not set. Wandb logging may fail.")

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Train emotion classification model')

parser.add_argument('--model_name', type=str, default="mental/mental-roberta-base", 
                    choices=["mental/mental-roberta-base", "mental/mental-bert-base-uncased", "answerdotai/ModernBERT-base"],
                    help='Model name or path (choose from available pre-trained models)')
parser.add_argument('--dataset_name', type=str, default="ESConv", 
                    choices=["ESConv"],
                    help='Dataset name (choose from available datasets)')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug mode (trains for only 3 epochs)')
parser.add_argument('--balanced', action='store_true',
                    help='Use balanced dataset')
parser.add_argument('--head_num', type=int, default=4,
                    help='Number of heads to use (default: 4)')
parser.add_argument('--model_metric', type=str, default="accuracy",
                    help='Metric to use for model evaluation (default: accuracy)')

args = parser.parse_args()

METRIC_MODEL = args.model_metric

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

# Base path for the model
data_path = f"/{args.dataset_name}/{model_abbreviation}"

# Create subfolder path based on arguments
if args.balanced:
    subfolder = f"{METRIC_MODEL}-{args.head_num}-heads-intensity-change-balanced"
else:
    subfolder = f"{METRIC_MODEL}-{args.head_num}-heads-intensity-change"

model_path = f"{data_path}/{subfolder}"

wandb.init(
    project="artificial-pain",
    name=f"{METRIC_MODEL}_{model_abbreviation}_{args.dataset_name}_{args.head_num}-heads-intensity-change{'_balanced' if args.balanced else ''}",
    config={
        "learning_rate": 1e-5,
        "epochs": 1 if args.debug else 1000,
        "batch_size": 6,
        "model": "EmotionModel",
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "patience": 50,
        "balanced": args.balanced,
        "head_num": args.head_num,
    }
)

# 3. Model Definition ----------------------------------------------
class EmotionClassifierHead(nn.Module):
    """Head for emotion type classification task."""
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        return self.fc2(x)


class IntensityRegressionHead(nn.Module):
    """Head for emotion intensity regression task."""
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.tanh = nn.Tanh()
        # One regressor output per emotion class
        self.fc2 = nn.Linear(128, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        return self.fc2(x)


class IntensityChangeRegressionHead(nn.Module):
    """Head for predicting the change in emotion intensity."""
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.tanh = nn.Tanh()
        # One regressor output per emotion class
        self.fc2 = nn.Linear(128, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        return self.fc2(x)


class Classifier(nn.Module):
    """
    Multi-task model for emotion classification and intensity/severity regression.
    Adapts to ESConv dataset.
    """
    def __init__(self, model_name, dataset_name):
        super().__init__()
        # Initialize the backbone model
        if 'ModernBERT' in model_name:
            self.encoding_backbone = ModernBertModel.from_pretrained(model_name)
        else:
            self.encoding_backbone = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.encoding_backbone.config.hidden_size
        
        # Set number of labels based on dataset
        if dataset_name == 'ESConv':
            self.num_labels = 7
            # Create heads for ESConv dataset
            self.classifier = EmotionClassifierHead(hidden_size, self.num_labels)
            self.initial_intensity_regressor = IntensityRegressionHead(hidden_size, self.num_labels)
            self.final_intensity_regressor = IntensityRegressionHead(hidden_size, self.num_labels)
            # Add new head for intensity change prediction
            self.intensity_change_regressor = IntensityChangeRegressionHead(hidden_size, self.num_labels)
            self.dataset_type = "ESConv"
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def forward(self, input_ids=None, attention_mask=None, labels=None, 
                initial_emotion_intensity=None, final_emotion_intensity=None, 
                intensity_change=None, avg_severity=None, return_dict=True):
        # Get backbone embeddings
        outputs = self.encoding_backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Forward pass through classifier head
        emotion_logits = self.classifier(cls_embedding)
        
        # Initialize losses and outputs
        loss = None
        classification_loss = None
        regression_outputs = {}
        regression_losses = {}
        
        # Dataset-specific regression heads
        if self.dataset_type == "ESConv":
            # Initial intensity regression
            initial_intensity_logits = self.initial_intensity_regressor(cls_embedding)
            regression_outputs['initial_intensity_logits'] = initial_intensity_logits
            
            # Final intensity regression
            final_intensity_logits = self.final_intensity_regressor(cls_embedding)
            regression_outputs['final_intensity_logits'] = final_intensity_logits
            
            # Intensity change regression
            intensity_change_logits = self.intensity_change_regressor(cls_embedding)
            regression_outputs['intensity_change_logits'] = intensity_change_logits
        
        # Calculate losses if labels are provided
        if labels is not None:
            # Classification loss
            classification_loss = nn.CrossEntropyLoss()(emotion_logits, labels)
            regression_losses['classification_loss'] = classification_loss
            
            # Create a mask for each sample based on its true label
            batch_size = labels.size(0)
            label_mask = torch.zeros(batch_size, self.num_labels, device=labels.device)
            for i in range(batch_size):
                # Check if the label is within the valid range before using it as an index
                if labels[i] >= 0 and labels[i] < self.num_labels:  # Ensure valid label
                    label_mask[i, labels[i]] = 1.0
                else:
                    print(f"Warning: Found invalid label {labels[i]} (expected 0-{self.num_labels-1})")
            
            # Dataset-specific regression losses
            if self.dataset_type == "ESConv":
                # Initial intensity loss
                if initial_emotion_intensity is not None:
                    # No need to mask the targets, just reshape them
                    initial_targets = initial_emotion_intensity.view(-1, 1)
                    
                    # Check if we have any valid targets (not NaN)
                    valid_mask = (label_mask.sum(dim=1) > 0) & ~torch.isnan(initial_targets.squeeze())

                    # Check if we have any valid targets
                    # Only compute loss if we have valid samples
                    if valid_mask.sum() > 0:
                        # Filter the data using the valid mask
                        filtered_logits = initial_intensity_logits[valid_mask]
                        filtered_targets = initial_targets[valid_mask]
                        filtered_label_mask = label_mask[valid_mask]
                        
                        # Compute loss only on valid samples
                        initial_intensity_loss = nn.MSELoss()(
                            filtered_logits[filtered_label_mask > 0],
                            filtered_targets.expand_as(filtered_logits)[filtered_label_mask > 0]
                        )
                        regression_losses['initial_intensity_loss'] = initial_intensity_loss
                
                # Final intensity loss
                if final_emotion_intensity is not None:
                    # No need to mask the targets, just reshape them
                    final_targets = final_emotion_intensity.view(-1, 1)
                    
                    # Check if we have any valid targets (not NaN)
                    valid_mask = (label_mask.sum(dim=1) > 0) & ~torch.isnan(final_targets.squeeze())
                    
                    # Only compute loss if we have valid samples
                    if valid_mask.sum() > 0:
                        # Filter the data using the valid mask
                        filtered_logits = final_intensity_logits[valid_mask]
                        filtered_targets = final_targets[valid_mask]
                        filtered_label_mask = label_mask[valid_mask]
                        
                        # Compute loss only on valid samples
                        final_intensity_loss = nn.MSELoss()(
                            filtered_logits[filtered_label_mask > 0],
                            filtered_targets.expand_as(filtered_logits)[filtered_label_mask > 0]
                        )
                        regression_losses['final_intensity_loss'] = final_intensity_loss
                
                # Intensity change loss
                if intensity_change is not None:
                    # Reshape intensity_change to match the expected dimensions
                    change_targets = intensity_change.view(-1, 1)
                    
                    # Check if we have any valid targets (not NaN)
                    valid_mask = (label_mask.sum(dim=1) > 0) & ~torch.isnan(change_targets.squeeze())
                    
                    # Only compute loss if we have valid samples
                    if valid_mask.sum() > 0:
                        # Filter the data using the valid mask
                        filtered_logits = intensity_change_logits[valid_mask]
                        filtered_targets = change_targets[valid_mask]
                        filtered_label_mask = label_mask[valid_mask]
                        
                        # Compute loss only on valid samples
                        intensity_change_loss = nn.MSELoss()(
                            filtered_logits[filtered_label_mask > 0],
                            filtered_targets.expand_as(filtered_logits)[filtered_label_mask > 0]
                        )
                        regression_losses['intensity_change_loss'] = intensity_change_loss
            
            # Combine all losses
            # Count how many regression losses we have (excluding classification_loss)
            num_regression_losses = sum(1 for name, value in regression_losses.items() 
                                       if name != 'classification_loss' and value is not None)

            # Set weights that add up to 1
            classification_weight = 0.5  # Adjust this value as needed
            regression_weight = 0.5 / num_regression_losses if num_regression_losses > 0 else 0

            # Apply weights
            loss = classification_weight * classification_loss
            for loss_name, loss_value in regression_losses.items():
                if loss_name != 'classification_loss' and loss_value is not None:
                    loss += regression_weight * loss_value
        
        if not return_dict:
            output = (emotion_logits,) + tuple(regression_outputs.values())
            return ((loss,) + output) if loss is not None else output
        
        return {
            'loss': loss,
            'logits': emotion_logits,
            **regression_outputs,
            **regression_losses
        }

# Define a custom data collator to handle the regression targets
class IntensityDataCollator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def __call__(self, features):
        # Extract all the necessary fields
        batch = {}
        
        # Standard fields for classification
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["labels"] = torch.tensor([f["labels"] for f in features])
        
        # Dataset-specific fields for regression
        if self.dataset_name == "ESConv":
            if "initial_emotion_intensity" in features[0]:
                batch["initial_emotion_intensity"] = torch.tensor([
                    self._convert_to_float(f["initial_emotion_intensity"]) 
                    for f in features
                ], dtype=torch.float)
            
            if "final_emotion_intensity" in features[0]:
                batch["final_emotion_intensity"] = torch.tensor([
                    self._convert_to_float(f["final_emotion_intensity"])
                    for f in features
                ], dtype=torch.float)
            
            # Add intensity_change calculation (final minus initial)
            if "initial_emotion_intensity" in features[0] and "final_emotion_intensity" in features[0]:
                batch["intensity_change"] = torch.tensor([
                    self._calculate_intensity_change(
                        self._convert_to_float(f["initial_emotion_intensity"]),
                        self._convert_to_float(f["final_emotion_intensity"])
                    )
                    for f in features
                ], dtype=torch.float)
                
        return batch
    
    def _convert_to_float(self, value):
        """Helper method to convert various value types to float."""
        if isinstance(value, torch.Tensor):
            return value.item()
        elif isinstance(value, (int, float)):
            return float(value) if not (isinstance(value, float) and math.isnan(value)) else float('nan')
        elif isinstance(value, str):
            if 'tensor' in value:
                # Extract the numeric value from tensor string
                try:
                    # Remove 'tensor(' and ')' and convert to float
                    cleaned = value.replace('tensor(', '').replace(')', '')
                    return float(cleaned)
                except:
                    return float('nan')
            else:
                try:
                    return float(value)
                except:
                    return float('nan')
        else:
            return float('nan')
    
    def _calculate_intensity_change(self, initial, final):
        """Calculate the change in intensity from initial to final."""
        if math.isnan(initial) or math.isnan(final):
            return float('nan')
        return final - initial  # Ensure this is final - initial consistently

# Create a custom trainer that handles the multi-task learning
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        
        Subclass and override for custom behavior.
        """
        # Forward pass
        outputs = model(**inputs)
        
        # Extract loss
        loss = outputs.get("loss")
        
        return (loss, outputs) if return_outputs else loss
        
    def create_optimizer(self):
        # Explicit parameter grouping
        encoding_backbone_params = []
        classifier_params = []
        regressor_params = []

        for name, param in self.model.named_parameters():
            if name.startswith("encoding_backbone"):
                encoding_backbone_params.append(param)
            elif name.startswith("classifier"):
                classifier_params.append(param)
            elif any(name.startswith(prefix) for prefix in ["initial_intensity_regressor", 
                                                           "final_intensity_regressor", 
                                                           "severity_regressor",
                                                           "intensity_change_regressor"]):
                regressor_params.append(param)
            else:
                raise ValueError(f"Unexpected parameter: {name}")

        optimizer_grouped_parameters = [
            {"params": encoding_backbone_params, "lr": 1e-5},
            {"params": classifier_params, "lr": 3e-5},
            {"params": regressor_params, "lr": 3e-5}  # Same learning rate as classifier
        ]

        self.optimizer = torch.optim.Adam(optimizer_grouped_parameters)
        return self.optimizer  # Make sure to return the optimizer
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Run the standard evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Get the current epoch
        epoch = self.state.epoch

        # Get the current learning rate
        learning_rate = self.optimizer.param_groups[0]['lr']

        # Get the patience counter from early stopping callback if it exists
        patience_counter = 0
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                # The EarlyStoppingCallback tracks a counter of epochs without improvement
                if hasattr(callback, 'patience'):
                    patience = callback.patience
                    # Different versions might use different attribute names
                    if hasattr(callback, 'early_stopping_patience_counter'):
                        patience_counter = patience - callback.early_stopping_patience_counter
                    elif hasattr(callback, 'patience_counter'):
                        patience_counter = patience - callback.patience_counter
                    else:
                        # If we can't find the counter, just use 0
                        patience_counter = 0

        # Safely access log history
        train_loss = 0
        train_accuracy = 0

        if len(self.state.log_history) > 1:
            last_train_log = self.state.log_history[-2]
            train_loss = last_train_log.get('loss', 0)
            train_accuracy = last_train_log.get('accuracy', 0)

        # Calculate regression metrics if we have regression outputs
        regression_metrics = {}
        
        # Get regression metrics from the evaluation outputs
        for key, value in output.items():
            if key.endswith('_loss') and key != 'eval_loss' and key != 'eval_classification_loss':
                # This is a regression loss
                regression_metrics[key] = value
        
        # Compute additional regression metrics (MSE, MAE, Pearson correlation)
        # We need to run a forward pass on the evaluation dataset to get predictions
        if hasattr(self.model, 'dataset_type'):
            # Get regression metrics for each emotion class
            from scipy.stats import pearsonr
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Create a dataloader for the evaluation dataset
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Initialize dictionaries to store predictions and targets
            initial_intensity_preds = {i: {'preds': [], 'true': []} for i in range(self.model.num_labels)}
            final_intensity_preds = {i: {'preds': [], 'true': []} for i in range(self.model.num_labels)}
            intensity_change_preds = {i: {'preds': [], 'true': []} for i in range(self.model.num_labels)}

            # Get the device from the model's parameters
            device = next(self.model.parameters()).device
            
            # Collect predictions and targets
            with torch.no_grad():
                for batch in eval_dataloader:
                    # Move batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    
                    # Get predictions and targets
                    labels = batch['labels']
                    
                    # Get predicted labels for evaluation
                    predicted_labels = torch.argmax(outputs['logits'], dim=1)
                    
                    # Dataset-specific processing
                    # Initial intensity
                    if 'initial_intensity_logits' in outputs and 'initial_emotion_intensity' in batch:
                        # For each sample in the batch
                        for i in range(len(labels)):
                            # Use TRUE label for evaluation, not predicted label
                            true_label = labels[i].item()
                            
                            # Store predictions for the TRUE emotion class
                            pred_intensity = outputs['initial_intensity_logits'][i, true_label].item()
                            initial_intensity_preds[true_label]['preds'].append(pred_intensity)
                            
                            # Store true value
                            if 'initial_emotion_intensity' in batch:
                                true_val = batch['initial_emotion_intensity'][i].item()
                                if not (isinstance(true_val, float) and math.isnan(true_val)):
                                    initial_intensity_preds[true_label]['true'].append(true_val)
                    
                    # Final intensity
                    if 'final_intensity_logits' in outputs and 'final_emotion_intensity' in batch:
                        # For each sample in the batch
                        for i in range(len(labels)):
                            # Use TRUE label for evaluation, not predicted label
                            true_label = labels[i].item()
                            
                            # Store predictions for the TRUE emotion class
                            pred_intensity = outputs['final_intensity_logits'][i, true_label].item()
                            final_intensity_preds[true_label]['preds'].append(pred_intensity)
                            
                            # Store true value
                            if 'final_emotion_intensity' in batch:
                                true_val = batch['final_emotion_intensity'][i].item()
                                if not (isinstance(true_val, float) and math.isnan(true_val)):
                                    final_intensity_preds[true_label]['true'].append(true_val)
                    
                    # Intensity change
                    if 'intensity_change_logits' in outputs and 'intensity_change' in batch:
                        # For each sample in the batch
                        for i in range(len(labels)):
                            # Use TRUE label for evaluation, not predicted label
                            true_label = labels[i].item()
                            
                            # Only process samples with valid intensity_change values
                            if 'intensity_change' in batch:
                                true_val = batch['intensity_change'][i].item()
                                if not (isinstance(true_val, float) and math.isnan(true_val)):
                                    # Store predictions for the TRUE emotion class
                                    pred_change = outputs['intensity_change_logits'][i, true_label].item()
                                    intensity_change_preds[true_label]['preds'].append(pred_change)
                                    intensity_change_preds[true_label]['true'].append(true_val)


            # Initialize lists for macro metrics
            all_mse_initial = []
            all_mae_initial = []
            all_pearson_initial = []
            all_mse_final = []
            all_mae_final = []
            all_pearson_final = []
            all_mse_change = []
            all_mae_change = []
            all_pearson_change = []
            all_weighted_mse_initial = []  # Add weighted MSE for initial intensity
            all_weighted_mse_final = []    # Add weighted MSE for final intensity
            all_weighted_mse_change = []   # Add weighted MSE for intensity change
            
            # For micro metrics - collect all predictions and true values across classes
            all_initial_preds = []
            all_initial_true = []
            all_final_preds = []
            all_final_true = []
            all_change_preds = []  # Add intensity change collections
            all_change_true = []  # Add intensity change collections
            all_initial_weights = []  # Add weights for initial intensity
            all_final_weights = []    # Add weights for final intensity
            all_change_weights = []   # Add weights for intensity change
            
            # Initial intensity
            for emotion_idx, data in initial_intensity_preds.items():
                if len(data['true']) >= 5:  # Only calculate if we have enough samples
                    preds = data['preds'][:len(data['true'])]
                    true = data['true']
                    
                    # Add to micro collections
                    all_initial_preds.extend(preds)
                    all_initial_true.extend(true)
                    # Add weights based on number of samples for this class
                    class_weight = len(true)
                    all_initial_weights.extend([class_weight] * len(true))
                    
                    from sklearn.metrics import mean_squared_error, mean_absolute_error
                    from scipy.stats import pearsonr
                    
                    mse = mean_squared_error(true, preds)
                    mae = mean_absolute_error(true, preds)
                    
                    # Calculate weighted MSE for this class
                    weighted_mse = mse * len(true)
                    
                    # Calculate Pearson correlation
                    try:
                        pearson_corr, p_value = pearsonr(true, preds)
                    except:
                        pearson_corr, p_value = float('nan'), float('nan')
                    
                    # Add to lists for macro averaging (only if not NaN)
                    if not math.isnan(mse):
                        all_mse_initial.append(mse)
                    if not math.isnan(mae):
                        all_mae_initial.append(mae)
                    if not math.isnan(pearson_corr):
                        all_pearson_initial.append(pearson_corr)
                    if not math.isnan(weighted_mse):
                        all_weighted_mse_initial.append(weighted_mse)
                    
                    regression_metrics[f'initial_intensity_emotion_{emotion_idx}_mse'] = mse
                    regression_metrics[f'initial_intensity_emotion_{emotion_idx}_mae'] = mae
                    regression_metrics[f'initial_intensity_emotion_{emotion_idx}_pearson'] = pearson_corr
                    regression_metrics[f'initial_intensity_emotion_{emotion_idx}_p_value'] = p_value
                    regression_metrics[f'initial_intensity_emotion_{emotion_idx}_samples'] = len(true)
                    regression_metrics[f'initial_intensity_emotion_{emotion_idx}_weighted_mse'] = weighted_mse
            
            # Final intensity
            for emotion_idx, data in final_intensity_preds.items():
                if len(data['true']) >= 5:  # Only calculate if we have enough samples
                    preds = data['preds'][:len(data['true'])]
                    true = data['true']
                    
                    # Add to micro collections
                    all_final_preds.extend(preds)
                    all_final_true.extend(true)
                    # Add weights based on number of samples for this class
                    class_weight = len(true)
                    all_final_weights.extend([class_weight] * len(true))
                    
                    from sklearn.metrics import mean_squared_error, mean_absolute_error
                    from scipy.stats import pearsonr
                    
                    mse = mean_squared_error(true, preds)
                    mae = mean_absolute_error(true, preds)
                    
                    # Calculate weighted MSE for this class
                    weighted_mse = mse * len(true)
                    
                    # Calculate Pearson correlation
                    try:
                        pearson_corr, p_value = pearsonr(true, preds)
                    except:
                        pearson_corr, p_value = float('nan'), float('nan')
                    
                    # Add to lists for macro averaging (only if not NaN)
                    if not math.isnan(mse):
                        all_mse_final.append(mse)
                    if not math.isnan(mae):
                        all_mae_final.append(mae)
                    if not math.isnan(pearson_corr):
                        all_pearson_final.append(pearson_corr)
                    if not math.isnan(weighted_mse):
                        all_weighted_mse_final.append(weighted_mse)
                    
                    regression_metrics[f'final_intensity_emotion_{emotion_idx}_mse'] = mse
                    regression_metrics[f'final_intensity_emotion_{emotion_idx}_mae'] = mae
                    regression_metrics[f'final_intensity_emotion_{emotion_idx}_pearson'] = pearson_corr
                    regression_metrics[f'final_intensity_emotion_{emotion_idx}_p_value'] = p_value
                    regression_metrics[f'final_intensity_emotion_{emotion_idx}_samples'] = len(true)
                    regression_metrics[f'final_intensity_emotion_{emotion_idx}_weighted_mse'] = weighted_mse
            
            # Intensity change
            for emotion_idx, data in intensity_change_preds.items():
                if len(data['true']) >= 5:  # Only calculate if we have enough samples
                    preds = data['preds'][:len(data['true'])]
                    true = data['true']
                    
                    # Add to micro collections
                    all_change_preds.extend(preds)
                    all_change_true.extend(true)
                    # Add weights based on number of samples for this class
                    class_weight = len(true)
                    all_change_weights.extend([class_weight] * len(true))
                    
                    from sklearn.metrics import mean_squared_error, mean_absolute_error
                    from scipy.stats import pearsonr
                    
                    mse = mean_squared_error(true, preds)
                    mae = mean_absolute_error(true, preds)
                    
                    # Calculate weighted MSE for this class
                    weighted_mse = mse * len(true)
                    
                    # Calculate Pearson correlation
                    try:
                        pearson_corr, p_value = pearsonr(true, preds)
                    except:
                        pearson_corr, p_value = float('nan'), float('nan')
                    
                    # Add to lists for macro averaging (only if not NaN)
                    if not math.isnan(mse):
                        all_mse_change.append(mse)
                    if not math.isnan(mae):
                        all_mae_change.append(mae)
                    if not math.isnan(pearson_corr):
                        all_pearson_change.append(pearson_corr)
                    if not math.isnan(weighted_mse):
                        all_weighted_mse_change.append(weighted_mse)
                    
                    regression_metrics[f'intensity_change_emotion_{emotion_idx}_mse'] = mse
                    regression_metrics[f'intensity_change_emotion_{emotion_idx}_mae'] = mae
                    regression_metrics[f'intensity_change_emotion_{emotion_idx}_pearson'] = pearson_corr
                    regression_metrics[f'intensity_change_emotion_{emotion_idx}_p_value'] = p_value
                    regression_metrics[f'intensity_change_emotion_{emotion_idx}_samples'] = len(true)
                    regression_metrics[f'intensity_change_emotion_{emotion_idx}_weighted_mse'] = weighted_mse
            
            # Calculate macro metrics for initial intensity
            if all_mse_initial:
                regression_metrics['initial_intensity_macro_mse'] = np.mean(all_mse_initial)
            if all_mae_initial:
                regression_metrics['initial_intensity_macro_mae'] = np.mean(all_mae_initial)
            if all_pearson_initial:
                regression_metrics['initial_intensity_macro_pearson'] = np.mean(all_pearson_initial)
            if all_weighted_mse_initial and sum(all_initial_weights) > 0:
                # Calculate weighted average MSE across classes
                regression_metrics['initial_intensity_macro_weighted_mse'] = sum(all_weighted_mse_initial) / sum(all_initial_weights)
            
            # Calculate macro metrics for final intensity
            if all_mse_final:
                regression_metrics['final_intensity_macro_mse'] = np.mean(all_mse_final)
            if all_mae_final:
                regression_metrics['final_intensity_macro_mae'] = np.mean(all_mae_final)
            if all_pearson_final:
                regression_metrics['final_intensity_macro_pearson'] = np.mean(all_pearson_final)
            if all_weighted_mse_final and sum(all_final_weights) > 0:
                # Calculate weighted average MSE across classes
                regression_metrics['final_intensity_macro_weighted_mse'] = sum(all_weighted_mse_final) / sum(all_final_weights)
            
            # Calculate macro metrics for final intensity
            if all_mse_change:
                regression_metrics['intensity_change_macro_mse'] = np.mean(all_mse_change)
            if all_mae_change:
                regression_metrics['intensity_change_macro_mae'] = np.mean(all_mae_change)
            if all_pearson_change:
                regression_metrics['intensity_change_macro_pearson'] = np.mean(all_pearson_change)
            if all_weighted_mse_change and sum(all_change_weights) > 0:
                # Calculate weighted average MSE across classes
                regression_metrics['intensity_change_macro_weighted_mse'] = sum(all_weighted_mse_change) / sum(all_change_weights)
            
            # Calculate micro metrics for initial intensity
            if len(all_initial_true) >= 5:
                regression_metrics['initial_intensity_micro_mse'] = mean_squared_error(all_initial_true, all_initial_preds)
                regression_metrics['initial_intensity_micro_mae'] = mean_absolute_error(all_initial_true, all_initial_preds)
                # Calculate weighted MSE using sample weights
                if all_initial_weights:
                    regression_metrics['initial_intensity_micro_weighted_mse'] = np.average(
                        [(true - pred)**2 for true, pred in zip(all_initial_true, all_initial_preds)],
                        weights=all_initial_weights
                    )
                try:
                    pearson_corr, p_value = pearsonr(all_initial_true, all_initial_preds)
                    regression_metrics['initial_intensity_micro_pearson'] = pearson_corr
                    regression_metrics['initial_intensity_micro_p_value'] = p_value
                except:
                    regression_metrics['initial_intensity_micro_pearson'] = float('nan')
                    regression_metrics['initial_intensity_micro_p_value'] = float('nan')
                regression_metrics['initial_intensity_micro_samples'] = len(all_initial_true)
            
            # Calculate micro metrics for final intensity
            if len(all_final_true) >= 5:
                regression_metrics['final_intensity_micro_mse'] = mean_squared_error(all_final_true, all_final_preds)
                regression_metrics['final_intensity_micro_mae'] = mean_absolute_error(all_final_true, all_final_preds)
                # Calculate weighted MSE using sample weights
                if all_final_weights:
                    regression_metrics['final_intensity_micro_weighted_mse'] = np.average(
                        [(true - pred)**2 for true, pred in zip(all_final_true, all_final_preds)],
                        weights=all_final_weights
                    )
                try:
                    pearson_corr, p_value = pearsonr(all_final_true, all_final_preds)
                    regression_metrics['final_intensity_micro_pearson'] = pearson_corr
                    regression_metrics['final_intensity_micro_p_value'] = p_value
                except:
                    regression_metrics['final_intensity_micro_pearson'] = float('nan')
                    regression_metrics['final_intensity_micro_p_value'] = float('nan')
                regression_metrics['final_intensity_micro_samples'] = len(all_final_true)
            
            # Calculate micro metrics for intensity change
            if len(all_change_true) >= 5:
                regression_metrics['intensity_change_micro_mse'] = mean_squared_error(all_change_true, all_change_preds)
                regression_metrics['intensity_change_micro_mae'] = mean_absolute_error(all_change_true, all_change_preds)
                # Calculate weighted MSE using sample weights
                if all_change_weights:
                    regression_metrics['intensity_change_micro_weighted_mse'] = np.average(
                        [(true - pred)**2 for true, pred in zip(all_change_true, all_change_preds)],
                        weights=all_change_weights
                    )
                try:
                    pearson_corr, p_value = pearsonr(all_change_true, all_change_preds)
                    regression_metrics['intensity_change_micro_pearson'] = pearson_corr
                    regression_metrics['intensity_change_micro_p_value'] = p_value
                except:
                    regression_metrics['intensity_change_micro_pearson'] = float('nan')
                    regression_metrics['intensity_change_micro_p_value'] = float('nan')
                regression_metrics['intensity_change_micro_samples'] = len(all_change_true)

        # Log metrics to wandb
        wandb_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_detection_loss": train_loss,
            "train_accuracy": train_accuracy,
            # Validation metrics are logged as eval_ in wandb console
            "val_loss": output.get(f"{metric_key_prefix}_loss", 0),  # Use get() with default value
            "val_detection_loss": output.get(f"{metric_key_prefix}_loss", 0),  # Use get() with default value
            "val_accuracy": output.get(f"{metric_key_prefix}_accuracy", 0),
            "val_macro_f1": output.get(f"{metric_key_prefix}_f1", 0),
            "val_recall": output.get(f"{metric_key_prefix}_recall", 0),
            "val_precision": output.get(f"{metric_key_prefix}_precision", 0),
            "learning_rate": learning_rate,
            "patience_counter": patience_counter
        }
        
        # Add regression metrics to wandb
        for metric_name, metric_value in regression_metrics.items():
            wandb_metrics[f"val_{metric_name}"] = metric_value
        
        wandb.log(wandb_metrics)

        return output

# 5. Evaluation Metrics --------------------------------------------
metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    """
    Compute metrics for both classification and regression tasks.
    """
    # Unpack the evaluation predictions
    if isinstance(eval_pred.predictions, tuple):
        # If predictions is a tuple, the first element is the classification logits
        classification_logits = eval_pred.predictions[0]
    else:
        # Otherwise, it's just the classification logits
        classification_logits = eval_pred.predictions
    
    labels = eval_pred.label_ids
    predictions = np.argmax(classification_logits, axis=-1)
    
    # Calculate standard micro accuracy (overall accuracy)
    micro_accuracy = {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}
    
    # Calculate macro accuracy (average of per-class accuracies)
    from sklearn.metrics import balanced_accuracy_score
    macro_accuracy = {"macro_accuracy": balanced_accuracy_score(labels, predictions)}
    
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    # Add recall calculation
    from sklearn.metrics import recall_score, precision_score
    recall = {"recall": recall_score(labels, predictions, average="macro")}
    precision = {"precision": precision_score(labels, predictions, average="macro")}
    
    # Combine all classification metrics
    metrics = {**micro_accuracy, **macro_accuracy, **f1, **recall, **precision}
    
    return metrics

# 6. Training Execution --------------------------------------------
# Add function to load and process the tokenized datasets
def load_tokenized_dataset(dataset_path):
    import pandas as pd
    import ast
    import torch
    
    # Load the CSV file
    df = pd.read_csv(dataset_path)
    
    # Convert string representations of tensors to actual tensors
    def parse_tensor(tensor_str):
        # Clean up the tensor string and convert to list of values
        if isinstance(tensor_str, str):
            # Remove tensor() wrapper and quotes
            cleaned = tensor_str.replace('tensor(', '').replace(')', '')
            # Handle single value tensors
            if ',' not in cleaned:
                return torch.tensor(int(cleaned))
            # Handle multi-dimensional tensors
            else:
                # Parse the string as a list
                try:
                    # Convert to list of integers
                    values = ast.literal_eval(cleaned)
                    return torch.tensor(values)
                except:
                    print(f"Error parsing: {tensor_str}")
                    return None
        return tensor_str
    
    # Apply the parsing to each column
    df['labels'] = df['labels'].apply(parse_tensor)
    df['input_ids'] = df['input_ids'].apply(parse_tensor)
    df['attention_mask'] = df['attention_mask'].apply(parse_tensor)
    
    # Convert to dataset format
    dataset = {
        'labels': df['labels'].tolist(),
        'input_ids': df['input_ids'].tolist(),
        'attention_mask': df['attention_mask'].tolist()
    }
    
    # Add intensity/severity data if available based on dataset type

    # Check if intensity columns exist in the dataframe with correct names
    if 'initial_emotion_intensity' in df.columns and 'final_emotion_intensity' in df.columns:
        dataset['initial_emotion_intensity'] = df['initial_emotion_intensity'].tolist()
        dataset['final_emotion_intensity'] = df['final_emotion_intensity'].tolist()
        print(f"Loaded emotion intensity data: {len(dataset['initial_emotion_intensity'])} initial and {len(dataset['final_emotion_intensity'])} final intensity values")
    else:
        print("Warning: ESConv dataset loaded but emotion intensity columns not found")
    
    return dataset

load_path = f"/{args.dataset_name}/{model_abbreviation}"
if args.balanced:
    train_dataset_path = f"{load_path}/balanced_tokenized_splits/train_split.csv"
    val_dataset_path = f"{load_path}/balanced_tokenized_splits/validation_split.csv"
    test_dataset_path = f"{load_path}/balanced_tokenized_splits/test_split.csv"
else:
    train_dataset_path = f"{load_path}/tokenized_splits/train_split.csv"
    val_dataset_path = f"{load_path}/tokenized_splits/validation_split.csv"
    test_dataset_path = f"{load_path}/tokenized_splits/test_split.csv"

train_dataset = load_tokenized_dataset(train_dataset_path)
val_dataset = load_tokenized_dataset(val_dataset_path)
test_dataset = load_tokenized_dataset(test_dataset_path)

# Create PyTorch datasets
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, initial_emotion_intensity=None, final_emotion_intensity=None, avg_severity=None):
        self.encodings = encodings
        self.labels = labels
        self.initial_emotion_intensity = initial_emotion_intensity
        self.final_emotion_intensity = final_emotion_intensity
        self.avg_severity = avg_severity
        self.dataset_name = args.dataset_name
        
        # Calculate intensity change if both initial and final intensities are available
        self.intensity_change = None
        if initial_emotion_intensity is not None and final_emotion_intensity is not None:
            self.intensity_change = []
            for i in range(len(initial_emotion_intensity)):
                initial = self._convert_to_float(initial_emotion_intensity[i])
                final = self._convert_to_float(final_emotion_intensity[i])
                if math.isnan(initial) or math.isnan(final):
                    self.intensity_change.append(float('nan'))
                else:
                    self.intensity_change.append(final - initial)  # Changed to final - initial to be consistent

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        
        # Add dataset-specific fields
        if self.initial_emotion_intensity is not None:
            item['initial_emotion_intensity'] = self.initial_emotion_intensity[idx]
        if self.final_emotion_intensity is not None:
            item['final_emotion_intensity'] = self.final_emotion_intensity[idx]
        if self.intensity_change is not None:
            item['intensity_change'] = self.intensity_change[idx]
            
        return item

    def __len__(self):
        return len(self.labels)
    
    def _convert_to_float(self, value):
        """Helper method to convert various value types to float."""
        if isinstance(value, torch.Tensor):
            return value.item()
        elif isinstance(value, (int, float)):
            return float(value) if not (isinstance(value, float) and math.isnan(value)) else float('nan')
        elif isinstance(value, str):
            if 'tensor' in value:
                # Extract the numeric value from tensor string
                try:
                    # Remove 'tensor(' and ')' and convert to float
                    cleaned = value.replace('tensor(', '').replace(')', '')
                    return float(cleaned)
                except:
                    return float('nan')
            else:
                try:
                    return float(value)
                except:
                    return float('nan')
        else:
            return float('nan')

# Create dataset objects with intensity/severity data based on dataset type
# For ESConv, include emotion intensity data
initial_emotion_intensity = train_dataset.get('initial_emotion_intensity', None)
final_emotion_intensity = train_dataset.get('final_emotion_intensity', None)

train_dataset_obj = TokenizedDataset(
    {'input_ids': train_dataset['input_ids'], 'attention_mask': train_dataset['attention_mask']},
    train_dataset['labels'],
    initial_emotion_intensity=initial_emotion_intensity,
    final_emotion_intensity=final_emotion_intensity
)

initial_emotion_intensity = val_dataset.get('initial_emotion_intensity', None)
final_emotion_intensity = val_dataset.get('final_emotion_intensity', None)

val_dataset_obj = TokenizedDataset(
    {'input_ids': val_dataset['input_ids'], 'attention_mask': val_dataset['attention_mask']},
    val_dataset['labels'],
    initial_emotion_intensity=initial_emotion_intensity,
    final_emotion_intensity=final_emotion_intensity
)

initial_emotion_intensity = test_dataset.get('initial_emotion_intensity', None)
final_emotion_intensity = test_dataset.get('final_emotion_intensity', None)

test_dataset_obj = TokenizedDataset(
    {'input_ids': test_dataset['input_ids'], 'attention_mask': test_dataset['attention_mask']},
    test_dataset['labels'],
    initial_emotion_intensity=initial_emotion_intensity,
    final_emotion_intensity=final_emotion_intensity
)

print("Successfully loaded tokenized datasets")

# Create data collator
data_collator = IntensityDataCollator(dataset_name=args.dataset_name)

# Initialize the model
model = Classifier(args.model_name, args.dataset_name)

# Define training arguments
training_args = TrainingArguments(
    output_dir=model_path,
    num_train_epochs=1 if args.debug else 1000,  # Use only 3 epochs in debug mode
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f"{model_path}/logs",
    logging_steps=10,
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model=METRIC_MODEL,
    greater_is_better=True if METRIC_MODEL == "accuracy" else False,
    save_total_limit=3,  # Only keep the 3 best checkpoints
    report_to="wandb",  # Report metrics to Weights & Biases
    fp16=torch.cuda.is_available(),  # Use mixed precision if available
)

# Initialize the trainer with the custom data collator
trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_obj,
    eval_dataset=val_dataset_obj,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=50)],  # Early stopping
    data_collator=data_collator,  # Use our custom data collator
)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")

# Save a reproducibility info file
def save_reproducibility_info(save_path):
    import json
    import torch
    import numpy as np
    import random
    import os

    # Collect reproducibility information
    info = {
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(device),
        "random_seed": 42,  # The seed used in your code
        "python_env": {
            "python_version": os.sys.version,
        }
    }

    # Save to JSON file
    with open(save_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"Reproducibility info saved to {save_path}")

# Save reproducibility information
reproducibility_path = f"{model_path}/reproducibility_info.json"
save_reproducibility_info(reproducibility_path)

# Add this before trainer.train()
print("Checking dataset for invalid labels...")
invalid_count = 0
for i, item in enumerate(train_dataset_obj):
    if item['labels'] < 0 or item['labels'] >= model.num_labels:
        print(f"Invalid label in training item {i}: {item['labels']}")
        invalid_count += 1
        if invalid_count > 10:  # Limit the number of errors to display
            print("Too many invalid labels, stopping check...")
            break
print(f"Found {invalid_count} invalid labels in training dataset")

# Start Training
trainer.train()

# 7. Final Evaluation ----------------------------------------------
print("\nFinal test evaluation:")
test_results = trainer.evaluate(test_dataset_obj, metric_key_prefix="test")
print(test_results)

# Save the best checkpoint to Google Drive
trainer.save_model(model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

tokenizer.save_pretrained(model_path)  # Also save the tokenizer

# Save model configuration for easier reloading
model_config = {
    "num_labels": model.num_labels,
    "base_model": args.model_name
}

import json
with open(f"{model_path}/model_config.json", "w") as f:
    json.dump(model_config, f)

from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, precision_score

# Log all relevant metrics to wandb at the end of training
wandb_metrics = {
    "test_loss": test_results.get('test_loss', 0),
}

# Add metrics that definitely exist
if 'test_accuracy' in test_results:
    wandb_metrics["test_micro_accuracy"] = test_results['test_accuracy']

# Add other metrics if they exist
for metric_name in ['f1', 'recall', 'precision', 'macro_accuracy']:
    test_key = f'test_{metric_name}'
    if test_key in test_results:
        # For f1, recall, and precision, these are macro by default
        if metric_name in ['f1', 'recall', 'precision']:
            wandb_metrics[f"test_macro_{metric_name}"] = test_results[test_key]
        else:
            wandb_metrics[test_key] = test_results[test_key]

# Log the metrics we were able to collect
wandb.log(wandb_metrics)

# Save wandb run information to a JSON file
# Check if wandb is initialized
if wandb.run is not None:
    # Get the current wandb run
    run = wandb.run

    # Collect all relevant wandb information
    wandb_info = {
        "run_id": run.id,
        "run_name": run.name,
        "run_path": f"{run.entity}/{run.project}/{run.id}",
        "run_url": run.get_url(),
        "project": run.project,
        "entity": run.entity,
        "group": run.group,
        "job_type": run.job_type,
        "config": {k: str(v) for k, v in run.config.items()},  # Convert config values to strings for JSON serialization
        "tags": run.tags,
        "notes": run.notes,
        "start_time": str(run.start_time),
        # "created_at": str(run.created_at)
    }

    # Define the save path
    wandb_info_path = f"{model_path}/wandb_info.json"

    # Save to JSON file
    with open(wandb_info_path, "w") as f:
        json.dump(wandb_info, f, indent=4)

    print(f"Wandb run information saved to {wandb_info_path}")
else:
    print("No active wandb run found. Make sure wandb is initialized.")

# Close wandb run
wandb.finish()

# Add function to compute inference metrics
def compute_inference_metrics(prediction_data, model):
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score
    from scipy.stats import pearsonr
    import math
    
    """
    Compute metrics using predicted labels instead of true labels.
    This simulates inference-time performance.
    """
    # Extract data
    true_labels = [pred["true_label"] for pred in prediction_data]
    predicted_labels = [pred["predicted_label"] for pred in prediction_data]
    
    # Calculate classification metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.metrics import balanced_accuracy_score
    
    
    micro_accuracy = accuracy_score(true_labels, predicted_labels)
    macro_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')
    micro_precision = precision_score(true_labels, predicted_labels, average='micro')
    macro_precision = precision_score(true_labels, predicted_labels, average='macro')
    weighted_precision = precision_score(true_labels, predicted_labels, average='weighted')
    micro_recall = recall_score(true_labels, predicted_labels, average='micro')
    macro_recall = recall_score(true_labels, predicted_labels, average='macro')
    weighted_recall = recall_score(true_labels, predicted_labels, average='weighted')
    # Calculate regression metrics
    regression_metrics = {}

    # Extract initial and final intensity data
    initial_data = {i: {'preds': [], 'true': []} for i in range(model.num_labels)}
    final_data = {i: {'preds': [], 'true': []} for i in range(model.num_labels)}
    change_data = {i: {'preds': [], 'true': []} for i in range(model.num_labels)}  # Add intensity change data
    
    for pred in prediction_data:
        if "initial_emotion_intensity_true" in pred and "initial_intensity_logits" in pred:
            # Use PREDICTED label for inference evaluation
            predicted_label = pred["predicted_label"]
            
            # Store prediction for the PREDICTED emotion class
            pred_intensity = pred["initial_intensity_logits"][predicted_label]
            initial_data[predicted_label]['preds'].append(pred_intensity)
            
            # Store true value
            true_intensity = pred["initial_emotion_intensity_true"]
            if not (isinstance(true_intensity, float) and math.isnan(true_intensity)):
                initial_data[predicted_label]['true'].append(true_intensity)
        
        if "final_emotion_intensity_true" in pred and "final_intensity_logits" in pred:
            # Use PREDICTED label for inference evaluation
            predicted_label = pred["predicted_label"]
            
            # Store prediction for the PREDICTED emotion class
            pred_intensity = pred["final_intensity_logits"][predicted_label]
            final_data[predicted_label]['preds'].append(pred_intensity)
            
            # Store true value
            true_intensity = pred["final_emotion_intensity_true"]
            if not (isinstance(true_intensity, float) and math.isnan(true_intensity)):
                final_data[predicted_label]['true'].append(true_intensity)

        # Add intensity change data collection
        if "intensity_change_true" in pred and "intensity_change_logits" in pred:
            # Only process if intensity_change_true is valid
            true_change = pred["intensity_change_true"]
            if not (isinstance(true_change, float) and math.isnan(true_change)):
                # Use PREDICTED label for inference evaluation
                predicted_label = pred["predicted_label"]
                
                # Store prediction for the PREDICTED emotion class
                pred_change = pred["intensity_change_logits"][predicted_label]
                change_data[predicted_label]['preds'].append(pred_change)
                change_data[predicted_label]['true'].append(true_change)
    
    # Calculate metrics for each emotion class
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    # Initialize lists for macro metrics
    all_mse_initial = []
    all_mae_initial = []
    all_pearson_initial = []
    all_mse_final = []
    all_mae_final = []
    all_pearson_final = []
    all_mse_change = []  # Add intensity change metrics
    all_mae_change = []  # Add intensity change metrics
    all_pearson_change = []  # Add intensity change metrics
    all_weighted_mse_initial = []  # Add weighted MSE for initial intensity
    all_weighted_mse_final = []    # Add weighted MSE for final intensity
    all_weighted_mse_change = []   # Add weighted MSE for intensity change
    
    # FOR micro metrics - collect all predictions and true values across classes
    all_initial_preds = []
    all_initial_true = []
    all_final_preds = []
    all_final_true = []
    all_change_preds = []  # Add intensity change collections
    all_change_true = []  # Add intensity change collections
    all_initial_weights = []  # Add weights for initial intensity
    all_final_weights = []    # Add weights for final intensity
    all_change_weights = []   # Add weights for intensity change
    
    # Initial intensity
    for emotion_idx, data in initial_data.items():
        if len(data['true']) >= 5:  # Only calculate if we have enough samples
            preds = data['preds'][:len(data['true'])]
            true = data['true']
            
            # Add to micro collections
            all_initial_preds.extend(preds)
            all_initial_true.extend(true)
            # Add weights based on number of samples for this class
            class_weight = len(true)
            all_initial_weights.extend([class_weight] * len(true))
            
            mse = mean_squared_error(true, preds)
            mae = mean_absolute_error(true, preds)
            
            # Calculate weighted MSE for this class
            weighted_mse = mse * len(true)
            
            # Calculate Pearson correlation
            try:
                pearson_corr, p_value = pearsonr(true, preds)
            except:
                pearson_corr, p_value = float('nan'), float('nan')
            
            # Add to lists for macro averaging (only if not NaN)
            if not math.isnan(mse):
                all_mse_initial.append(mse)
            if not math.isnan(mae):
                all_mae_initial.append(mae)
            if not math.isnan(pearson_corr):
                all_pearson_initial.append(pearson_corr)
            if not math.isnan(weighted_mse):
                all_weighted_mse_initial.append(weighted_mse)
            
            regression_metrics[f'initial_intensity_emotion_{emotion_idx}_mse'] = mse
            regression_metrics[f'initial_intensity_emotion_{emotion_idx}_mae'] = mae
            regression_metrics[f'initial_intensity_emotion_{emotion_idx}_pearson'] = pearson_corr
            regression_metrics[f'initial_intensity_emotion_{emotion_idx}_p_value'] = p_value
            regression_metrics[f'initial_intensity_emotion_{emotion_idx}_samples'] = len(true)
            regression_metrics[f'initial_intensity_emotion_{emotion_idx}_weighted_mse'] = weighted_mse
    
    # Final intensity
    for emotion_idx, data in final_data.items():
        if len(data['true']) >= 5:  # Only calculate if we have enough samples
            preds = data['preds'][:len(data['true'])]
            true = data['true']
            
            # Add to micro collections
            all_final_preds.extend(preds)
            all_final_true.extend(true)
            # Add weights based on number of samples for this class
            class_weight = len(true)
            all_final_weights.extend([class_weight] * len(true))
            
            mse = mean_squared_error(true, preds)
            mae = mean_absolute_error(true, preds)
            
            # Calculate weighted MSE for this class
            weighted_mse = mse * len(true)
            
            # Calculate Pearson correlation
            try:
                pearson_corr, p_value = pearsonr(true, preds)
            except:
                pearson_corr, p_value = float('nan'), float('nan')
            
            # Add to lists for macro averaging (only if not NaN)
            if not math.isnan(mse):
                all_mse_final.append(mse)
            if not math.isnan(mae):
                all_mae_final.append(mae)
            if not math.isnan(pearson_corr):
                all_pearson_final.append(pearson_corr)
            if not math.isnan(weighted_mse):
                all_weighted_mse_final.append(weighted_mse)
            
            regression_metrics[f'final_intensity_emotion_{emotion_idx}_mse'] = mse
            regression_metrics[f'final_intensity_emotion_{emotion_idx}_mae'] = mae
            regression_metrics[f'final_intensity_emotion_{emotion_idx}_pearson'] = pearson_corr
            regression_metrics[f'final_intensity_emotion_{emotion_idx}_p_value'] = p_value
            regression_metrics[f'final_intensity_emotion_{emotion_idx}_samples'] = len(true)
            regression_metrics[f'final_intensity_emotion_{emotion_idx}_weighted_mse'] = weighted_mse
    
    for emotion_idx, data in change_data.items():
        if len(data['true']) >= 5:  # Only calculate if we have enough samples
            preds = data['preds'][:len(data['true'])]
            true = data['true']
            
            # Add to micro collections
            all_change_preds.extend(preds)
            all_change_true.extend(true)
            # Add weights based on number of samples for this class
            class_weight = len(true)
            all_change_weights.extend([class_weight] * len(true))
            
            mse = mean_squared_error(true, preds)
            mae = mean_absolute_error(true, preds)
            
            # Calculate weighted MSE for this class
            weighted_mse = mse * len(true)
            
            # Calculate Pearson correlation
            try:
                pearson_corr, p_value = pearsonr(true, preds)
            except:
                pearson_corr, p_value = float('nan'), float('nan')
            
            # Add to lists for macro averaging (only if not NaN)
            if not math.isnan(mse):
                all_mse_change.append(mse)
            if not math.isnan(mae):
                all_mae_change.append(mae)
            if not math.isnan(pearson_corr):
                all_pearson_change.append(pearson_corr)
            if not math.isnan(weighted_mse):
                all_weighted_mse_change.append(weighted_mse)
            
            regression_metrics[f'intensity_change_emotion_{emotion_idx}_mse'] = mse
            regression_metrics[f'intensity_change_emotion_{emotion_idx}_mae'] = mae
            regression_metrics[f'intensity_change_emotion_{emotion_idx}_pearson'] = pearson_corr
            regression_metrics[f'intensity_change_emotion_{emotion_idx}_p_value'] = p_value
            regression_metrics[f'intensity_change_emotion_{emotion_idx}_samples'] = len(true)
            regression_metrics[f'intensity_change_emotion_{emotion_idx}_weighted_mse'] = weighted_mse

    # Calculate macro metrics for initial intensity
    if all_mse_initial:
        regression_metrics['initial_intensity_macro_mse'] = np.mean(all_mse_initial)
    if all_mae_initial:
        regression_metrics['initial_intensity_macro_mae'] = np.mean(all_mae_initial)
    if all_pearson_initial:
        regression_metrics['initial_intensity_macro_pearson'] = np.mean(all_pearson_initial)
    if all_weighted_mse_initial and sum(all_initial_weights) > 0:
        # Calculate weighted average MSE across classes
        regression_metrics['initial_intensity_macro_weighted_mse'] = sum(all_weighted_mse_initial) / sum(all_initial_weights)
    
    # Calculate macro metrics for final intensity
    if all_mse_final:
        regression_metrics['final_intensity_macro_mse'] = np.mean(all_mse_final)
    if all_mae_final:
        regression_metrics['final_intensity_macro_mae'] = np.mean(all_mae_final)
    if all_pearson_final:
        regression_metrics['final_intensity_macro_pearson'] = np.mean(all_pearson_final)
    if all_weighted_mse_final and sum(all_final_weights) > 0:
        # Calculate weighted average MSE across classes
        regression_metrics['final_intensity_macro_weighted_mse'] = sum(all_weighted_mse_final) / sum(all_final_weights)
    
    # Calculate macro metrics for final intensity
    if all_mse_change:
        regression_metrics['intensity_change_macro_mse'] = np.mean(all_mse_change)
    if all_mae_change:
        regression_metrics['intensity_change_macro_mae'] = np.mean(all_mae_change)
    if all_pearson_change:
        regression_metrics['intensity_change_macro_pearson'] = np.mean(all_pearson_change)
    if all_weighted_mse_change and sum(all_change_weights) > 0:
        # Calculate weighted average MSE across classes
        regression_metrics['intensity_change_macro_weighted_mse'] = sum(all_weighted_mse_change) / sum(all_change_weights)
    
    # Calculate micro metrics for initial intensity
    if len(all_initial_true) >= 5:
        regression_metrics['initial_intensity_micro_mse'] = mean_squared_error(all_initial_true, all_initial_preds)
        regression_metrics['initial_intensity_micro_mae'] = mean_absolute_error(all_initial_true, all_initial_preds)
        # Calculate weighted MSE using sample weights
        if all_initial_weights:
            regression_metrics['initial_intensity_micro_weighted_mse'] = np.average(
                [(true - pred)**2 for true, pred in zip(all_initial_true, all_initial_preds)],
                weights=all_initial_weights
            )
        try:
            pearson_corr, p_value = pearsonr(all_initial_true, all_initial_preds)
            regression_metrics['initial_intensity_micro_pearson'] = pearson_corr
            regression_metrics['initial_intensity_micro_p_value'] = p_value
        except:
            regression_metrics['initial_intensity_micro_pearson'] = float('nan')
            regression_metrics['initial_intensity_micro_p_value'] = float('nan')
        regression_metrics['initial_intensity_micro_samples'] = len(all_initial_true)
    
    # Calculate micro metrics for final intensity
    if len(all_final_true) >= 5:
        regression_metrics['final_intensity_micro_mse'] = mean_squared_error(all_final_true, all_final_preds)
        regression_metrics['final_intensity_micro_mae'] = mean_absolute_error(all_final_true, all_final_preds)
        # Calculate weighted MSE using sample weights
        if all_final_weights:
            regression_metrics['final_intensity_micro_weighted_mse'] = np.average(
                [(true - pred)**2 for true, pred in zip(all_final_true, all_final_preds)],
                weights=all_final_weights
            )
        try:
            pearson_corr, p_value = pearsonr(all_final_true, all_final_preds)
            regression_metrics['final_intensity_micro_pearson'] = pearson_corr
            regression_metrics['final_intensity_micro_p_value'] = p_value
        except:
            regression_metrics['final_intensity_micro_pearson'] = float('nan')
            regression_metrics['final_intensity_micro_p_value'] = float('nan')
        regression_metrics['final_intensity_micro_samples'] = len(all_final_true)
    
    # Calculate micro metrics for intensity change
    if len(all_change_true) >= 5:
        regression_metrics['intensity_change_micro_mse'] = mean_squared_error(all_change_true, all_change_preds)
        regression_metrics['intensity_change_micro_mae'] = mean_absolute_error(all_change_true, all_change_preds)
        # Calculate weighted MSE using sample weights
        if all_change_weights:
            regression_metrics['intensity_change_micro_weighted_mse'] = np.average(
                [(true - pred)**2 for true, pred in zip(all_change_true, all_change_preds)],
                weights=all_change_weights
            )
        try:
            pearson_corr, p_value = pearsonr(all_change_true, all_change_preds)
            regression_metrics['intensity_change_micro_pearson'] = pearson_corr
            regression_metrics['intensity_change_micro_p_value'] = p_value
        except:
            regression_metrics['intensity_change_micro_pearson'] = float('nan')
            regression_metrics['intensity_change_micro_p_value'] = float('nan')
        regression_metrics['intensity_change_micro_samples'] = len(all_change_true)

    # Combine all metrics
    metrics = {
        'micro_accuracy': micro_accuracy,
        'macro_accuracy': macro_accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'micro_precision': micro_precision,
        'macro_precision': macro_precision,
        'weighted_precision': weighted_precision,
        'micro_recall': micro_recall,
        'macro_recall': macro_recall,
        'weighted_recall': weighted_recall,
        **regression_metrics
    }
    
    return metrics

# Evaluation script without using HuggingFace Trainer
print("\nEvaluating model without Trainer:")

def evaluate_model_without_trainer(model, dataset, device, split_name, save_dir):
    import numpy as np  # Add this import
    import torch
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    import math
    import json
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    # For regression evaluation - store predictions and true values for each emotion class
    regression_data = {}
    regression_data = {
        'initial_intensity': {i: {'preds': [], 'true': []} for i in range(model.num_labels)},
        'final_intensity': {i: {'preds': [], 'true': []} for i in range(model.num_labels)},
        'intensity_change': {i: {'preds': [], 'true': []} for i in range(model.num_labels)}
    }

    # Helper function to convert tensor strings to float values
    def convert_to_float(value):
        if isinstance(value, torch.Tensor):
            return value.item()
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str) and 'tensor' in value:
            # Extract the numeric value from tensor string
            try:
                # Remove 'tensor(' and ')' and convert to float
                cleaned = value.replace('tensor(', '').replace(')', '')
                return float(cleaned)
            except:
                print(f"Could not convert tensor string: {value}")
                return float('nan')
        else:
            try:
                return float(value)
            except:
                print(f"Could not convert to float: {value}")
                return float('nan')

    # Process one example at a time to avoid batch issues
    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]  # Get example by index, not by key

            # Extract and prepare inputs
            if 'input_ids' in example:
                # Try different approaches based on the type
                if isinstance(example['input_ids'], torch.Tensor):
                    input_ids = example['input_ids'].unsqueeze(0).to(device)
                    attention_mask = example['attention_mask'].unsqueeze(0).to(device)
                elif isinstance(example['input_ids'], list):
                    input_ids = torch.tensor([example['input_ids']]).to(device)
                    attention_mask = torch.tensor([example['attention_mask']]).to(device)
                else:
                    print(f"Unexpected type for input_ids: {type(example['input_ids'])}")
                    continue

                # Get label
                if 'labels' in example:
                    label = example['labels']
                elif 'label' in example:
                    label = example['label']
                else:
                    print("No label found in example")
                    continue

                # Convert label to tensor if needed
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label)
                
                # Get the true label as an integer
                true_label = label.item() if isinstance(label, torch.Tensor) else label

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']

                # Get prediction
                prediction = torch.argmax(logits, dim=-1).item()

                # Store prediction and label
                all_predictions.append(prediction)
                all_labels.append(true_label)
                
                # Store regression predictions based on TRUE label (not predicted label)
                # Initial intensity
                if 'initial_intensity_logits' in outputs and 'initial_emotion_intensity' in example:
                    # Get the predicted intensity for the TRUE emotion class
                    pred_intensity = outputs['initial_intensity_logits'][0, true_label].item()
                    regression_data['initial_intensity'][true_label]['preds'].append(pred_intensity)
                    
                    # Store true value
                    true_intensity = convert_to_float(example['initial_emotion_intensity'])
                    if not math.isnan(true_intensity):
                        regression_data['initial_intensity'][true_label]['true'].append(true_intensity)
                
                # Final intensity
                if 'final_intensity_logits' in outputs and 'final_emotion_intensity' in example:
                    # Get the predicted intensity for the TRUE emotion class
                    pred_intensity = outputs['final_intensity_logits'][0, true_label].item()
                    regression_data['final_intensity'][true_label]['preds'].append(pred_intensity)
                    
                    # Store true value
                    true_intensity = convert_to_float(example['final_emotion_intensity'])
                    if not math.isnan(true_intensity):
                        regression_data['final_intensity'][true_label]['true'].append(true_intensity)
                
                # Intensity change - fix to handle consistently with other metrics
                if 'intensity_change_logits' in outputs and 'intensity_change' in example:
                    # Only process if intensity_change is valid
                    true_change = convert_to_float(example['intensity_change'])
                    if not math.isnan(true_change):
                        # Get the predicted change for the TRUE emotion class
                        pred_change = outputs['intensity_change_logits'][0, true_label].item()
                        regression_data['intensity_change'][true_label]['preds'].append(pred_change)
                        regression_data['intensity_change'][true_label]['true'].append(true_change)

    # Calculate classification metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    
    micro_accuracy = accuracy_score(all_labels, all_predictions)
    macro_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
    micro_precision = precision_score(all_labels, all_predictions, average='micro')
    macro_precision = precision_score(all_labels, all_predictions, average='macro')
    micro_recall = recall_score(all_labels, all_predictions, average='micro')
    macro_recall = recall_score(all_labels, all_predictions, average='macro')
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Calculate regression metrics for each emotion class
    regression_metrics = {}
    
    # Add the weighted MSE metrics to the output
    if 'initial_intensity_macro_weighted_mse' in regression_metrics:
        metrics['initial_intensity_macro_weighted_mse'] = regression_metrics['initial_intensity_macro_weighted_mse']
    if 'final_intensity_macro_weighted_mse' in regression_metrics:
        metrics['final_intensity_macro_weighted_mse'] = regression_metrics['final_intensity_macro_weighted_mse']
    if 'intensity_change_macro_weighted_mse' in regression_metrics:
        metrics['intensity_change_macro_weighted_mse'] = regression_metrics['intensity_change_macro_weighted_mse']
    if 'initial_intensity_micro_weighted_mse' in regression_metrics:
        metrics['initial_intensity_micro_weighted_mse'] = regression_metrics['initial_intensity_micro_weighted_mse']
    if 'final_intensity_micro_weighted_mse' in regression_metrics:
        metrics['final_intensity_micro_weighted_mse'] = regression_metrics['final_intensity_micro_weighted_mse']
    if 'intensity_change_micro_weighted_mse' in regression_metrics:
        metrics['intensity_change_micro_weighted_mse'] = regression_metrics['intensity_change_micro_weighted_mse']

    # Initialize lists to store metrics for macro averaging
    all_mse_initial = []
    all_mae_initial = []
    all_pearson_initial = []
    all_mse_final = []
    all_mae_final = []
    all_pearson_final = []
    all_mse_change = []
    all_mae_change = []
    all_pearson_change = []
    
    # For micro metrics - collect all predictions and true values across classes
    all_initial_preds = []
    all_initial_true = []
    all_final_preds = []
    all_final_true = []
    all_change_preds = []
    all_change_true = []
    
    # Initial intensity
    for emotion_idx, data in regression_data['initial_intensity'].items():
        if len(data['true']) >= 5:  # Only calculate if we have enough samples
            preds = data['preds'][:len(data['true'])]
            true = data['true']
            
            # Add to micro collections
            all_initial_preds.extend(preds)
            all_initial_true.extend(true)
            
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from scipy.stats import pearsonr
            
            mse = mean_squared_error(true, preds)
            mae = mean_absolute_error(true, preds)
            
            # Calculate Pearson correlation
            try:
                pearson_corr, p_value = pearsonr(true, preds)
            except:
                pearson_corr, p_value = float('nan'), float('nan')
            
            # Add to lists for macro averaging (only if not NaN)
            if not math.isnan(mse):
                all_mse_initial.append(mse)
            if not math.isnan(mae):
                all_mae_initial.append(mae)
            if not math.isnan(pearson_corr):
                all_pearson_initial.append(pearson_corr)
            
            regression_metrics[f'initial_intensity_emotion_{emotion_idx}_mse'] = mse
            regression_metrics[f'initial_intensity_emotion_{emotion_idx}_mae'] = mae
            regression_metrics[f'initial_intensity_emotion_{emotion_idx}_pearson'] = pearson_corr
            regression_metrics[f'initial_intensity_emotion_{emotion_idx}_p_value'] = p_value
            regression_metrics[f'initial_intensity_emotion_{emotion_idx}_samples'] = len(true)
    
    # Final intensity
    for emotion_idx, data in regression_data['final_intensity'].items():
        if len(data['true']) >= 5:  # Only calculate if we have enough samples
            preds = data['preds'][:len(data['true'])]
            true = data['true']
            
            # Add to micro collections
            all_final_preds.extend(preds)
            all_final_true.extend(true)
            
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from scipy.stats import pearsonr
            
            mse = mean_squared_error(true, preds)
            mae = mean_absolute_error(true, preds)
            
            # Calculate Pearson correlation
            try:
                pearson_corr, p_value = pearsonr(true, preds)
            except:
                pearson_corr, p_value = float('nan'), float('nan')
            
            # Add to lists for macro averaging (only if not NaN)
            if not math.isnan(mse):
                all_mse_final.append(mse)
            if not math.isnan(mae):
                all_mae_final.append(mae)
            if not math.isnan(pearson_corr):
                all_pearson_final.append(pearson_corr)
            
            regression_metrics[f'final_intensity_emotion_{emotion_idx}_mse'] = mse
            regression_metrics[f'final_intensity_emotion_{emotion_idx}_mae'] = mae
            regression_metrics[f'final_intensity_emotion_{emotion_idx}_pearson'] = pearson_corr
            regression_metrics[f'final_intensity_emotion_{emotion_idx}_p_value'] = p_value
            regression_metrics[f'final_intensity_emotion_{emotion_idx}_samples'] = len(true)
    
    # Intensity change
    for emotion_idx, data in regression_data['intensity_change'].items():
        if len(data['true']) >= 5:  # Only calculate if we have enough samples
            preds = data['preds'][:len(data['true'])]
            true = data['true']
            
            # Add to micro collections
            all_change_preds.extend(preds)
            all_change_true.extend(true)
            
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from scipy.stats import pearsonr
            
            mse = mean_squared_error(true, preds)
            mae = mean_absolute_error(true, preds)
            
            # Calculate Pearson correlation
            try:
                pearson_corr, p_value = pearsonr(true, preds)
            except:
                pearson_corr, p_value = float('nan'), float('nan')
            
            # Add to lists for macro averaging (only if not NaN)
            if not math.isnan(mse):
                all_mse_change.append(mse)
            if not math.isnan(mae):
                all_mae_change.append(mae)
            if not math.isnan(pearson_corr):
                all_pearson_change.append(pearson_corr)
            
            regression_metrics[f'intensity_change_emotion_{emotion_idx}_mse'] = mse
            regression_metrics[f'intensity_change_emotion_{emotion_idx}_mae'] = mae
            regression_metrics[f'intensity_change_emotion_{emotion_idx}_pearson'] = pearson_corr
            regression_metrics[f'intensity_change_emotion_{emotion_idx}_p_value'] = p_value
            regression_metrics[f'intensity_change_emotion_{emotion_idx}_samples'] = len(true)
    
    # Calculate macro metrics for initial intensity
    if all_mse_initial:
        regression_metrics['initial_intensity_macro_mse'] = np.mean(all_mse_initial)
    if all_mae_initial:
        regression_metrics['initial_intensity_macro_mae'] = np.mean(all_mae_initial)
    if all_pearson_initial:
        regression_metrics['initial_intensity_macro_pearson'] = np.mean(all_pearson_initial)
    
    # Calculate macro metrics for final intensity
    if all_mse_final:
        regression_metrics['final_intensity_macro_mse'] = np.mean(all_mse_final)
    if all_mae_final:
        regression_metrics['final_intensity_macro_mae'] = np.mean(all_mae_final)
    if all_pearson_final:
        regression_metrics['final_intensity_macro_pearson'] = np.mean(all_pearson_final)
    
    # Calculate macro metrics for final intensity
    if all_mse_change:
        regression_metrics['intensity_change_macro_mse'] = np.mean(all_mse_change)
    if all_mae_change:
        regression_metrics['intensity_change_macro_mae'] = np.mean(all_mae_change)
    if all_pearson_change:
        regression_metrics['intensity_change_macro_pearson'] = np.mean(all_pearson_change)
    
    # Calculate micro metrics for initial intensity
    if len(all_initial_true) >= 5:
        regression_metrics['initial_intensity_micro_mse'] = mean_squared_error(all_initial_true, all_initial_preds)
        regression_metrics['initial_intensity_micro_mae'] = mean_absolute_error(all_initial_true, all_initial_preds)
        try:
            pearson_corr, p_value = pearsonr(all_initial_true, all_initial_preds)
            regression_metrics['initial_intensity_micro_pearson'] = pearson_corr
            regression_metrics['initial_intensity_micro_p_value'] = p_value
        except:
            regression_metrics['initial_intensity_micro_pearson'] = float('nan')
            regression_metrics['initial_intensity_micro_p_value'] = float('nan')
        regression_metrics['initial_intensity_micro_samples'] = len(all_initial_true)
    
    # Calculate micro metrics for final intensity
    if len(all_final_true) >= 5:
        regression_metrics['final_intensity_micro_mse'] = mean_squared_error(all_final_true, all_final_preds)
        regression_metrics['final_intensity_micro_mae'] = mean_absolute_error(all_final_true, all_final_preds)
        try:
            pearson_corr, p_value = pearsonr(all_final_true, all_final_preds)
            regression_metrics['final_intensity_micro_pearson'] = pearson_corr
            regression_metrics['final_intensity_micro_p_value'] = p_value
        except:
            regression_metrics['final_intensity_micro_pearson'] = float('nan')
            regression_metrics['final_intensity_micro_p_value'] = float('nan')
        regression_metrics['final_intensity_micro_samples'] = len(all_final_true)
    
    # Calculate micro metrics for intensity change
    if len(all_change_true) >= 5:
        regression_metrics['intensity_change_micro_mse'] = mean_squared_error(all_change_true, all_change_preds)
        regression_metrics['intensity_change_micro_mae'] = mean_absolute_error(all_change_true, all_change_preds)
        try:
            pearson_corr, p_value = pearsonr(all_change_true, all_change_preds)
            regression_metrics['intensity_change_micro_pearson'] = pearson_corr
            regression_metrics['intensity_change_micro_p_value'] = p_value
        except:
            regression_metrics['intensity_change_micro_pearson'] = float('nan')
            regression_metrics['intensity_change_micro_p_value'] = float('nan')
        regression_metrics['intensity_change_micro_samples'] = len(all_change_true)

    # Print metrics
    print(f"\n{split_name} Classification Metrics:")
    print(f"Micro Accuracy: {micro_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    
    # Calculate and save additional metrics for classification
    from sklearn.metrics import classification_report
    
    # Generate classification report
    target_names = [f"Class {i}" for i in range(model.num_labels)]
    class_report = classification_report(all_labels, all_predictions, 
                                        target_names=target_names, 
                                        output_dict=True)
    
    # Calculate macro accuracy
    macro_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    
    # Save normalized confusion matrix
    norm_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Save as PNG instead of NPY
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(norm_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Normalized Confusion Matrix - {split_name}')
    plt.colorbar()
    
    # Add labels and ticks
    tick_marks = np.arange(model.num_labels)
    plt.xticks(tick_marks, [f'Class {i}' for i in range(model.num_labels)], rotation=45)
    plt.yticks(tick_marks, [f'Class {i}' for i in range(model.num_labels)])
    
    # Add text annotations
    thresh = norm_conf_matrix.max() / 2.
    for i in range(norm_conf_matrix.shape[0]):
        for j in range(norm_conf_matrix.shape[1]):
            plt.text(j, i, f'{norm_conf_matrix[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if norm_conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the figure
    conf_matrix_path = f"{save_dir}/{split_name}_norm_confusion_matrix.png"
    plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Extract per-class metrics
    per_class_metrics = {}
    for class_name in target_names:
        per_class_metrics[class_name] = {
            'precision': class_report[class_name]['precision'],
            'recall': class_report[class_name]['recall'],
            'f1': class_report[class_name]['f1-score'],
            'support': class_report[class_name]['support']
        }
    
    # Prepare comprehensive metrics dictionary
    comprehensive_metrics = {
        'micro_accuracy': micro_accuracy,
        'macro_accuracy': macro_accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'micro_recall': micro_recall,
        'per_class_metrics': per_class_metrics,
        'classification_report': class_report,
        'confusion_matrix_path': conf_matrix_path,
        **regression_metrics
    }
    
    import json

    # Save comprehensive metrics to file
    with open(f"{save_dir}/{split_name}_comprehensive_metrics.json", 'w') as f:
        json.dump({k: v for k, v in comprehensive_metrics.items() if k != 'classification_report'}, f, indent=4)
    
    # Save classification report separately (it's a nested dict)
    with open(f"{save_dir}/{split_name}_classification_report.json", 'w') as f:
        json.dump(class_report, f, indent=4)
    
    # Save detailed predictions for test set
    if split_name == "test":
        # Collect all predictions in a structured format
        all_prediction_data = []
        
        # Process one example at a time to collect detailed predictions
        model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                example = dataset[i]
                
                # Extract and prepare inputs
                if 'input_ids' in example:
                    # Try different approaches based on the type
                    if isinstance(example['input_ids'], torch.Tensor):
                        input_ids = example['input_ids'].unsqueeze(0).to(device)
                        attention_mask = example['attention_mask'].unsqueeze(0).to(device)
                    elif isinstance(example['input_ids'], list):
                        input_ids = torch.tensor([example['input_ids']]).to(device)
                        attention_mask = torch.tensor([example['attention_mask']]).to(device)
                    else:
                        continue
                    
                    # Get label
                    if 'labels' in example:
                        label = example['labels']
                    elif 'label' in example:
                        label = example['label']
                    else:
                        continue
                    
                    # Convert label to tensor if needed
                    if not isinstance(label, torch.Tensor):
                        label = torch.tensor(label)
                    
                    # Get the true label as an integer
                    true_label = label.item() if isinstance(label, torch.Tensor) else label
                    
                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']
                    
                    # Get prediction
                    predicted_label = torch.argmax(logits, dim=-1).item()
                    
                    # Create a prediction record
                    prediction_record = {
                        "example_id": i,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "logits": logits[0].cpu().numpy().tolist(),
                    }
                    
                    # Add dataset-specific intensity predictions for ALL emotion classes
                    # Get all intensity predictions for all emotion classes
                    if 'initial_intensity_logits' in outputs:
                        prediction_record["initial_intensity_logits"] = outputs['initial_intensity_logits'][0].cpu().numpy().tolist()
                        # Add ground truth intensity if available
                        if 'initial_emotion_intensity' in example:
                            true_intensity = convert_to_float(example['initial_emotion_intensity'])
                            prediction_record["initial_emotion_intensity_true"] = true_intensity
                    
                    if 'final_intensity_logits' in outputs:
                        prediction_record["final_intensity_logits"] = outputs['final_intensity_logits'][0].cpu().numpy().tolist()
                        # Add ground truth intensity if available
                        if 'final_emotion_intensity' in example:
                            true_intensity = convert_to_float(example['final_emotion_intensity'])
                            prediction_record["final_emotion_intensity_true"] = true_intensity
                    
                    if 'intensity_change_logits' in outputs:
                        prediction_record["intensity_change_logits"] = outputs['intensity_change_logits'][0].cpu().numpy().tolist()
                        # Calculate ground truth intensity change if initial and final intensities are available
                        if 'initial_emotion_intensity' in example and 'final_emotion_intensity' in example:
                            initial_intensity = convert_to_float(example['initial_emotion_intensity'])
                            final_intensity = convert_to_float(example['final_emotion_intensity'])
                            if not (math.isnan(initial_intensity) or math.isnan(final_intensity)):
                                true_change = final_intensity - initial_intensity
                                prediction_record["intensity_change_true"] = true_change
                    
                    all_prediction_data.append(prediction_record)
        
        # Save detailed predictions to file
        import pandas as pd
        predictions_df = pd.DataFrame(all_prediction_data)
        predictions_path = f"{save_dir}/{split_name}_detailed_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Detailed predictions saved to {predictions_path}")
        
        # Also save as JSON for easier programmatic access
        import json
        json_path = f"{save_dir}/{split_name}_detailed_predictions.json"
        with open(json_path, 'w') as f:
            json.dump(all_prediction_data, f, indent=2)
        print(f"Detailed predictions also saved to {json_path}")
        
        # Create an additional "inference" evaluation if this is the test set
        if split_name == "test":
            # Create inference metrics using predicted labels instead of true labels
            inference_metrics = compute_inference_metrics(all_prediction_data, model)
            
            # Add inference metrics to comprehensive metrics
            comprehensive_metrics["inference"] = inference_metrics
            
            # Save inference metrics to a separate file with clear micro/macro labeling
            with open(f"{save_dir}/inference_comprehensive_metrics.json", 'w') as f:
                json.dump(inference_metrics, f, indent=4)
            
            print(f"Inference metrics saved to {save_dir}/inference_comprehensive_metrics.json")
        
    # Return the comprehensive metrics
    return comprehensive_metrics

# Use the already loaded tokenized datasets
validation_dataset = val_dataset_obj
test_dataset = test_dataset_obj
print("Successfully loaded and tokenized datasets")

from safetensors.torch import load_file

# Load the model
model = Classifier(args.model_name, args.dataset_name)
state_dict = load_file(f"{model_path}/model.safetensors")
model.load_state_dict(state_dict)
model.to(device)

import os

# Make sure the model path directory exists
os.makedirs(model_path, exist_ok=True)

# Helper function to organize metrics with clear micro/macro labeling
def organize_metrics(metrics_dict):
    organized = {
        "classification": {
            "micro": {},
            "macro": {},
            "per_class": {}
        },
        "regression": {
            "micro": {},
            "macro": {},
            "per_class": {}
        }
    }
    
    # Process classification metrics
    if "micro_accuracy" in metrics_dict:
        organized["classification"]["micro"]["accuracy"] = metrics_dict["micro_accuracy"]
    if "micro_f1" in metrics_dict:
        organized["classification"]["micro"]["f1"] = metrics_dict["micro_f1"]
    if "micro_precision" in metrics_dict:
        organized["classification"]["micro"]["precision"] = metrics_dict["micro_precision"]
    if "micro_recall" in metrics_dict:
        organized["classification"]["micro"]["recall"] = metrics_dict["micro_recall"]
    
    if "macro_accuracy" in metrics_dict:
        organized["classification"]["macro"]["accuracy"] = metrics_dict["macro_accuracy"]
    if "macro_f1" in metrics_dict:
        organized["classification"]["macro"]["f1"] = metrics_dict["macro_f1"]
    if "macro_precision" in metrics_dict:
        organized["classification"]["macro"]["precision"] = metrics_dict["macro_precision"]
    if "macro_recall" in metrics_dict:
        organized["classification"]["macro"]["recall"] = metrics_dict["macro_recall"]
    
    # Add per-class classification metrics
    if "per_class_metrics" in metrics_dict:
        organized["classification"]["per_class"] = metrics_dict["per_class_metrics"]
    
    # Process regression metrics
    for key, value in metrics_dict.items():
        # Skip non-regression metrics
        if not any(x in key for x in ['mse', 'mae', 'pearson', 'p_value', 'samples']):
            continue
        
        # Determine if this is a micro, macro, or per-class metric
        if 'micro' in key:
            # Extract the metric type (e.g., initial_intensity, final_intensity, severity)
            parts = key.split('_')
            metric_type = '_'.join(parts[:-2])  # e.g., 'initial_intensity' or 'severity'
            metric_name = parts[-1]  # e.g., 'mse', 'mae', 'pearson'
            
            if metric_type not in organized["regression"]["micro"]:
                organized["regression"]["micro"][metric_type] = {}
            organized["regression"]["micro"][metric_type][metric_name] = value
            
        elif 'macro' in key:
            # Extract the metric type
            parts = key.split('_')
            metric_type = '_'.join(parts[:-2])  # e.g., 'initial_intensity' or 'severity'
            metric_name = parts[-1]  # e.g., 'mse', 'mae', 'pearson'
            
            if metric_type not in organized["regression"]["macro"]:
                organized["regression"]["macro"][metric_type] = {}
            organized["regression"]["macro"][metric_type][metric_name] = value
            
        elif 'emotion_' in key:
            # This is a per-class metric
            parts = key.split('_')
            metric_type = '_'.join(parts[:-3])  # e.g., 'initial_intensity' or 'severity'
            class_idx = parts[-2]  # e.g., '0', '1', etc.
            metric_name = parts[-1]  # e.g., 'mse', 'mae', 'pearson'
            
            if metric_type not in organized["regression"]["per_class"]:
                organized["regression"]["per_class"][metric_type] = {}
            if class_idx not in organized["regression"]["per_class"][metric_type]:
                organized["regression"]["per_class"][metric_type][class_idx] = {}
            organized["regression"]["per_class"][metric_type][class_idx][metric_name] = value
    
    # Add classification report if available
    if "classification_report" in metrics_dict:
        organized["classification"]["report"] = metrics_dict["classification_report"]
    
    # Add confusion matrix path if available
    if "confusion_matrix_path" in metrics_dict:
        organized["classification"]["confusion_matrix_path"] = metrics_dict["confusion_matrix_path"]
    
    return organized


# Evaluate on validation set
val_metrics = evaluate_model_without_trainer(model, validation_dataset, device, 
                                            split_name="validation", save_dir=model_path)
print(f"Validation metrics (without Trainer):")
print(f"  Micro Accuracy: {val_metrics['micro_accuracy']:.4f}")
print(f"  Macro Accuracy: {val_metrics['macro_accuracy']:.4f}")
print(f"  Macro F1 Score: {val_metrics['macro_f1']:.4f}")
print(f"  Macro Recall: {val_metrics['macro_recall']:.4f}")
print(f"  Macro Precision: {val_metrics['macro_precision']:.4f}")
print(f"  Micro F1 Score: {val_metrics['micro_f1']:.4f}")
print(f"  Micro Recall: {val_metrics['micro_recall']:.4f}")
print(f"  Micro Precision: {val_metrics['micro_precision']:.4f}")

print("\nPer-class metrics (validation):")
for class_name, metrics in val_metrics['per_class_metrics'].items():
    print(f"  {class_name}:")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    F1: {metrics['f1']:.4f}")

print(f"\nNormalized Confusion Matrix saved to: {val_metrics['confusion_matrix_path']}")

# More organized display of regression metrics
print("\nRegression metrics (validation):")
# Group metrics by emotion and metric type
emotion_metrics = {}
for metric_name, metric_value in val_metrics.items():
    if any(x in metric_name for x in ['mse', 'mae', 'pearson', 'samples']) and isinstance(metric_value, (int, float)):
        parts = metric_name.split('_')
        if 'macro' in metric_name:
            # Handle macro metrics separately
            emotion_type = '_'.join(parts[:-2])  # e.g., 'initial_intensity' or 'severity'
            metric_type = parts[-1]  # e.g., 'mse', 'mae', 'pearson'
            key = f"{emotion_type}_macro"
            if key not in emotion_metrics:
                emotion_metrics[key] = {}
            emotion_metrics[key][metric_type] = metric_value
        elif len(parts) >= 3:
            emotion_type = '_'.join(parts[:-2])  # e.g., 'initial_intensity' or 'severity'
            emotion_idx = parts[-2]  # e.g., 'emotion_0'
            metric_type = parts[-1]  # e.g., 'mse', 'mae', 'pearson'
            
            key = f"{emotion_type}_{emotion_idx}"
            if key not in emotion_metrics:
                emotion_metrics[key] = {}
            emotion_metrics[key][metric_type] = metric_value

# Print macro metrics first
print("  Macro Metrics:")
for emotion_key, metrics in emotion_metrics.items():
    if 'macro' in emotion_key:
        print(f"  {emotion_key}:")
        for metric_type, value in metrics.items():
            print(f"    {metric_type}: {value:.4f}")

# Print micro metrics
print("\n  Micro Metrics:")
for emotion_key, metrics in emotion_metrics.items():
    if 'micro' in emotion_key:
        print(f"  {emotion_key}:")
        for metric_type, value in metrics.items():
            print(f"    {metric_type}: {value:.4f}")

# Print per-class metrics
print("\n  Per-class Metrics:")
for emotion_key, metrics in emotion_metrics.items():
    if 'macro' not in emotion_key and 'micro' not in emotion_key:
        print(f"  {emotion_key}:")
        for metric_type, value in metrics.items():
            print(f"    {metric_type}: {value:.4f}")

# Evaluate on test set
test_metrics = evaluate_model_without_trainer(model, test_dataset, device, 
                                             split_name="test", save_dir=model_path)
print(f"\nTest metrics (without Trainer):")
print(f"  Micro Accuracy: {test_metrics['micro_accuracy']:.4f}")
print(f"  Macro Accuracy: {test_metrics['macro_accuracy']:.4f}")
print(f"  Macro F1 Score: {test_metrics['macro_f1']:.4f}")
print(f"  Macro Recall: {test_metrics['macro_recall']:.4f}")
print(f"  Macro Precision: {test_metrics['macro_precision']:.4f}")
print(f"  Micro F1 Score: {test_metrics['micro_f1']:.4f}")
print(f"  Micro Recall: {test_metrics['micro_recall']:.4f}")
print(f"  Micro Precision: {test_metrics['micro_precision']:.4f}")

print("\nPer-class metrics (test):")
for class_name, metrics in test_metrics['per_class_metrics'].items():
    print(f"  {class_name}:")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    F1: {metrics['f1']:.4f}")

print(f"\nNormalized Confusion Matrix saved to: {test_metrics['confusion_matrix_path']}")

# More organized display of regression metrics
print("\nRegression metrics (test):")
# Group metrics by emotion and metric type
emotion_metrics = {}
for metric_name, metric_value in test_metrics.items():
    if any(x in metric_name for x in ['mse', 'mae', 'pearson', 'samples']) and isinstance(metric_value, (int, float)):
        parts = metric_name.split('_')
        if 'macro' in metric_name:
            # Handle macro metrics separately
            emotion_type = '_'.join(parts[:-2])  # e.g., 'initial_intensity' or 'severity'
            metric_type = parts[-1]  # e.g., 'mse', 'mae', 'pearson'
            key = f"{emotion_type}_macro"
            if key not in emotion_metrics:
                emotion_metrics[key] = {}
            emotion_metrics[key][metric_type] = metric_value
        elif len(parts) >= 3:
            emotion_type = '_'.join(parts[:-2])  # e.g., 'initial_intensity' or 'severity'
            emotion_idx = parts[-2]  # e.g., 'emotion_0'
            metric_type = parts[-1]  # e.g., 'mse', 'mae', 'pearson'
            
            key = f"{emotion_type}_{emotion_idx}"
            if key not in emotion_metrics:
                emotion_metrics[key] = {}
            emotion_metrics[key][metric_type] = metric_value

# Print macro metrics first
print("  Macro Metrics:")
for emotion_key, metrics in emotion_metrics.items():
    if 'macro' in emotion_key:
        print(f"  {emotion_key}:")
        for metric_type, value in metrics.items():
            print(f"    {metric_type}: {value:.4f}")

# Print micro metrics
print("\n  Micro Metrics:")
for emotion_key, metrics in emotion_metrics.items():
    if 'micro' in emotion_key:
        print(f"  {emotion_key}:")
        for metric_type, value in metrics.items():
            print(f"    {metric_type}: {value:.4f}")

# Print per-class metrics
print("\n  Per-class Metrics:")
for emotion_key, metrics in emotion_metrics.items():
    if 'macro' not in emotion_key and 'micro' not in emotion_key:
        print(f"  {emotion_key}:")
        for metric_type, value in metrics.items():
            print(f"    {metric_type}: {value:.4f}")

# Save results with clear micro/macro labeling
import json
results = {
    "validation": organize_metrics(val_metrics),
    "test": organize_metrics(test_metrics)
}

# Add inference results if they exist
if "inference" in test_metrics:
    results["inference"] = organize_metrics(test_metrics["inference"])

# Save organized results
with open(f"{model_path}/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {model_path}/evaluation_results.json")

# Create a summary file with key metrics from all evaluation modes
summary = {
    "validation": {
        "classification": {
            "micro": {
                "accuracy": val_metrics.get("micro_accuracy", 0),
                "f1": val_metrics.get("micro_f1", 0)
            },
            "macro": {
                "f1": val_metrics.get("macro_f1", 0)
            }
        },
        "regression": {
            "macro": {},
            "micro": {}
        }
    },
    "test": {
        "classification": {
            "micro": {
                "accuracy": test_metrics.get("micro_accuracy", 0),
                "f1": test_metrics.get("micro_f1", 0)
            },
            "macro": {
                "f1": test_metrics.get("macro_f1", 0)
            }
        },
        "regression": {
            "macro": {},
            "micro": {}
        }
    }
}

# Add regression metrics to summary with clear micro/macro labeling
for mode in ["validation", "test"]:
    metrics = val_metrics if mode == "validation" else test_metrics
    
    # Initialize regression sections if they don't exist
    if "regression" not in summary[mode]:
        summary[mode]["regression"] = {"micro": {}, "macro": {}}
    
    # Add dataset-specific regression metrics
    # Initial intensity - macro metrics
    if "initial_intensity_macro_pearson" in metrics:
        if "initial_intensity" not in summary[mode]["regression"]["macro"]:
            summary[mode]["regression"]["macro"]["initial_intensity"] = {}
        summary[mode]["regression"]["macro"]["initial_intensity"]["pearson"] = metrics.get("initial_intensity_macro_pearson", 0)
        summary[mode]["regression"]["macro"]["initial_intensity"]["mse"] = metrics.get("initial_intensity_macro_mse", 0)
        summary[mode]["regression"]["macro"]["initial_intensity"]["mae"] = metrics.get("initial_intensity_macro_mae", 0)
    
    # Initial intensity - micro metrics
    if "initial_intensity_micro_pearson" in metrics:
        if "initial_intensity" not in summary[mode]["regression"]["micro"]:
            summary[mode]["regression"]["micro"]["initial_intensity"] = {}
        summary[mode]["regression"]["micro"]["initial_intensity"]["pearson"] = metrics.get("initial_intensity_micro_pearson", 0)
        summary[mode]["regression"]["micro"]["initial_intensity"]["mse"] = metrics.get("initial_intensity_micro_mse", 0)
        summary[mode]["regression"]["micro"]["initial_intensity"]["mae"] = metrics.get("initial_intensity_micro_mae", 0)
    
    # Final intensity - macro metrics
    if "final_intensity_macro_pearson" in metrics:
        if "final_intensity" not in summary[mode]["regression"]["macro"]:
            summary[mode]["regression"]["macro"]["final_intensity"] = {}
        summary[mode]["regression"]["macro"]["final_intensity"]["pearson"] = metrics.get("final_intensity_macro_pearson", 0)
        summary[mode]["regression"]["macro"]["final_intensity"]["mse"] = metrics.get("final_intensity_macro_mse", 0)
        summary[mode]["regression"]["macro"]["final_intensity"]["mae"] = metrics.get("final_intensity_macro_mae", 0)
    
    # Final intensity - micro metrics
    if "final_intensity_micro_pearson" in metrics:
        if "final_intensity" not in summary[mode]["regression"]["micro"]:
            summary[mode]["regression"]["micro"]["final_intensity"] = {}
        summary[mode]["regression"]["micro"]["final_intensity"]["pearson"] = metrics.get("final_intensity_micro_pearson", 0)
        summary[mode]["regression"]["micro"]["final_intensity"]["mse"] = metrics.get("final_intensity_micro_mse", 0)
        summary[mode]["regression"]["micro"]["final_intensity"]["mae"] = metrics.get("final_intensity_micro_mae", 0)

# Add inference metrics to summary if they exist
if "inference" in test_metrics:
    inference_metrics = test_metrics["inference"]
    
    # Initialize inference section if it doesn't exist
    if "inference" not in summary:
        summary["inference"] = {
            "classification": {
                "micro": {
                    "accuracy": inference_metrics.get("micro_accuracy", 0),
                    "f1": inference_metrics.get("micro_f1", 0)
                },
                "macro": {
                    "f1": inference_metrics.get("macro_f1", 0)
                }
            },
            "regression": {
                "micro": {},
                "macro": {}
            }
        }
    
    # Add dataset-specific regression metrics for inference
    # Initial intensity - macro
    if "initial_intensity_macro_pearson" in inference_metrics:
        if "initial_intensity" not in summary["inference"]["regression"]["macro"]:
            summary["inference"]["regression"]["macro"]["initial_intensity"] = {}
        summary["inference"]["regression"]["macro"]["initial_intensity"]["pearson"] = inference_metrics.get("initial_intensity_macro_pearson", 0)
        summary["inference"]["regression"]["macro"]["initial_intensity"]["mse"] = inference_metrics.get("initial_intensity_macro_mse", 0)
        summary["inference"]["regression"]["macro"]["initial_intensity"]["mae"] = inference_metrics.get("initial_intensity_macro_mae", 0)
    
    # Initial intensity - micro
    if "initial_intensity_micro_pearson" in inference_metrics:
        if "initial_intensity" not in summary["inference"]["regression"]["micro"]:
            summary["inference"]["regression"]["micro"]["initial_intensity"] = {}
        summary["inference"]["regression"]["micro"]["initial_intensity"]["pearson"] = inference_metrics.get("initial_intensity_micro_pearson", 0)
        summary["inference"]["regression"]["micro"]["initial_intensity"]["mse"] = inference_metrics.get("initial_intensity_micro_mse", 0)
        summary["inference"]["regression"]["micro"]["initial_intensity"]["mae"] = inference_metrics.get("initial_intensity_micro_mae", 0)
    
    # Final intensity - macro
    if "final_intensity_macro_pearson" in inference_metrics:
        if "final_intensity" not in summary["inference"]["regression"]["macro"]:
            summary["inference"]["regression"]["macro"]["final_intensity"] = {}
        summary["inference"]["regression"]["macro"]["final_intensity"]["pearson"] = inference_metrics.get("final_intensity_macro_pearson", 0)
        summary["inference"]["regression"]["macro"]["final_intensity"]["mse"] = inference_metrics.get("final_intensity_macro_mse", 0)
        summary["inference"]["regression"]["macro"]["final_intensity"]["mae"] = inference_metrics.get("final_intensity_macro_mae", 0)
    
    # Final intensity - micro
    if "final_intensity_micro_pearson" in inference_metrics:
        if "final_intensity" not in summary["inference"]["regression"]["micro"]:
            summary["inference"]["regression"]["micro"]["final_intensity"] = {}
        summary["inference"]["regression"]["micro"]["final_intensity"]["pearson"] = inference_metrics.get("final_intensity_micro_pearson", 0)
        summary["inference"]["regression"]["micro"]["final_intensity"]["mse"] = inference_metrics.get("final_intensity_micro_mse", 0)
        summary["inference"]["regression"]["micro"]["final_intensity"]["mae"] = inference_metrics.get("final_intensity_micro_mae", 0)
    
    # Intensity change - macro metrics
    if "intensity_change_macro_pearson" in metrics:
        if "intensity_change" not in summary[mode]["regression"]["macro"]:
            summary[mode]["regression"]["macro"]["intensity_change"] = {}
        summary[mode]["regression"]["macro"]["intensity_change"]["pearson"] = metrics.get("intensity_change_macro_pearson", 0)
        summary[mode]["regression"]["macro"]["intensity_change"]["mse"] = metrics.get("intensity_change_macro_mse", 0)
        summary[mode]["regression"]["macro"]["intensity_change"]["mae"] = metrics.get("intensity_change_macro_mae", 0)
    
    # Intensity change - micro metrics
    if "intensity_change_micro_pearson" in metrics:
        if "intensity_change" not in summary[mode]["regression"]["micro"]:
            summary[mode]["regression"]["micro"]["intensity_change"] = {}
        summary[mode]["regression"]["micro"]["intensity_change"]["pearson"] = metrics.get("intensity_change_micro_pearson", 0)
        summary[mode]["regression"]["micro"]["intensity_change"]["mse"] = metrics.get("intensity_change_micro_mse", 0)
        summary[mode]["regression"]["micro"]["intensity_change"]["mae"] = metrics.get("intensity_change_micro_mae", 0)
    
import json
# Save summary
with open(f"{model_path}/evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=4)
print(f"Evaluation summary saved to {model_path}/evaluation_summary.json")
# Evaluate on test set
test_metrics = evaluate_model_without_trainer(model, test_dataset, device, 
                                             split_name="test", save_dir=model_path)

# Extract prediction data from test metrics to compute inference metrics
if "all_prediction_data" in test_metrics:
    inference_metrics = compute_inference_metrics(test_metrics["all_prediction_data"], model)
    # Add inference metrics to test_metrics
    test_metrics["inference"] = inference_metrics
    
    # Print inference metrics
    print(f"\nInference metrics (using predicted labels):")
    print(f"  Micro Accuracy: {inference_metrics['micro_accuracy']:.4f}")
    print(f"  Macro F1 Score: {inference_metrics['macro_f1']:.4f}")
    print(f"  Macro Recall: {inference_metrics['macro_recall']:.4f}")
    print(f"  Macro Precision: {inference_metrics['macro_precision']:.4f}")
    
    # Print regression metrics for inference mode
    print("\nRegression metrics (inference):")
    # Group metrics by emotion and metric type
    emotion_metrics = {}
    for metric_name, metric_value in inference_metrics.items():
        if any(x in metric_name for x in ['mse', 'mae', 'pearson', 'samples']) and isinstance(metric_value, (int, float)):
            parts = metric_name.split('_')
            if 'macro' in metric_name:
                # Handle macro metrics separately
                emotion_type = '_'.join(parts[:-2])  # e.g., 'initial_intensity' or 'severity'
                metric_type = parts[-1]  # e.g., 'mse', 'mae', 'pearson'
                key = f"{emotion_type}_macro"
                if key not in emotion_metrics:
                    emotion_metrics[key] = {}
                emotion_metrics[key][metric_type] = metric_value
            elif len(parts) >= 3:
                emotion_type = '_'.join(parts[:-2])  # e.g., 'initial_intensity' or 'severity'
                emotion_idx = parts[-2]  # e.g., 'emotion_0'
                metric_type = parts[-1]  # e.g., 'mse', 'mae', 'pearson'
                
                key = f"{emotion_type}_{emotion_idx}"
                if key not in emotion_metrics:
                    emotion_metrics[key] = {}
                emotion_metrics[key][metric_type] = metric_value
    
    # Print macro metrics first
    print("  Macro Metrics:")
    for emotion_key, metrics in emotion_metrics.items():
        if 'macro' in emotion_key:
            print(f"  {emotion_key}:")
            for metric_type, value in metrics.items():
                print(f"    {metric_type}: {value:.4f}")
    
    # Print micro metrics
    print("\n  Micro Metrics:")
    for emotion_key, metrics in emotion_metrics.items():
        if 'micro' in emotion_key:
            print(f"  {emotion_key}:")
            for metric_type, value in metrics.items():
                print(f"    {metric_type}: {value:.4f}")