# Used for all MentalRoBERTa, MentalBERT, and ModernBERT 

# Command to run script: 
# python classification-model.py --model_name mental/mental-roberta-base --dataset_name ESConv

import subprocess
try:
    subprocess.check_call(["pip", "install", "transformers", "datasets", "evaluate", "torch", "scikit-learn", "pandas", "openpyxl", "wandb", "argparse", "huggingface_hub", "safetensors", "matplotlib", "seaborn"])
except:
    print("Package installation failed or packages already installed")

import wandb
import evaluate
import argparse
import random

import json
import os
import torch
from safetensors.torch import load_file
from torch import nn
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, EarlyStoppingCallback, ModernBertModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MODEL_METRIC = "loss"

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
                    choices=["ESConv", "SAD"],
                    help='Dataset name (choose from available datasets)')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug mode (trains for only 3 epochs)')
parser.add_argument('--balanced', action='store_true',
                    help='Use balanced dataset for training and evaluation')

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

# Update model path to include "balanced" when the flag is set
if args.balanced:
    model_path = f"{args.dataset_name}/{model_abbreviation}/{MODEL_METRIC}-classification-balanced"
else:
    model_path = f"{args.dataset_name}/{model_abbreviation}/{MODEL_METRIC}-classification"

wandb.init(
    project="artificial-pain",
    name=f"{MODEL_METRIC}_{model_abbreviation}_{args.dataset_name}_{'balanced' if args.balanced else 'unbalanced'}_classification",
    config={
        "learning_rate": 1e-5,
        "epochs": 1 if args.debug else 1000,
        "batch_size": 6,
        "model": "EmotionModel",
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "patience": 50,
    }
)

# 3. Model Definition ----------------------------------------------
class Classifier(nn.Module):
    def __init__(self, model_name, dataset_name):
        super().__init__()
        if 'ModernBERT' in model_name:
            self.encoding_backbone = ModernBertModel.from_pretrained(model_name)
        else:
            self.encoding_backbone = AutoModel.from_pretrained(model_name)

        if dataset_name == 'ESConv':
            self.num_labels = 7

        elif dataset_name == 'SAD':
            self.num_labels = 9
        
        self.classifier = nn.Sequential(
            nn.Linear(self.encoding_backbone.config.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, self.num_labels)  
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoding_backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}

model = Classifier(args.model_name, args.dataset_name)

# 4. Training Setup ------------------------------------------------
# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

training_args = TrainingArguments(
    output_dir=f"{model_path}/results",
    eval_strategy="epoch",
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=1 if args.debug else 1000,  # Use only 3 epochs in debug mode
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir=f"{model_path}/logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model=MODEL_METRIC,
    fp16=(device.type == "cuda"),  # Enable mixed precision training only on CUDA
    report_to="wandb",
    save_total_limit=2,  # Only keep the 2 most recent checkpoints
    save_safetensors=True,  # Use the more reliable safetensors format
)

class CustomTrainer(Trainer):
    def create_optimizer(self):
        # Explicit parameter grouping
        encoding_backbone_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if name.startswith("encoding_backbone"):
                encoding_backbone_params.append(param)
            elif name.startswith("classifier"):
                classifier_params.append(param)
            else:
                raise ValueError(f"Unexpected parameter: {name}")

        optimizer_grouped_parameters = [
            {"params": encoding_backbone_params, "lr": 1e-5},
            {"params": classifier_params, "lr": 3e-5}
        ]

        return torch.optim.Adam(optimizer_grouped_parameters)
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Call `create_optimizer` to create the optimizer then `create_scheduler` to create the scheduler.
        """
        self.optimizer = self.create_optimizer() # create the optimizer first, then pass it to create_scheduler
        # By default, the Hugging Face Trainer will call create_scheduler during create_optimizer_and_scheduler.
        # Thus we only need to assign the self.optimizer attribute.
        Trainer.create_optimizer_and_scheduler(self, num_training_steps)

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

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_detection_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": output.get(f"{metric_key_prefix}_loss", 0),  # Use get() with default value
            "val_detection_loss": output.get(f"{metric_key_prefix}_loss", 0),  # Use get() with default value
            "val_accuracy": output.get(f"{metric_key_prefix}_accuracy", 0),
            "val_macro_f1": output.get(f"{metric_key_prefix}_f1", 0),
            "val_recall": output.get(f"{metric_key_prefix}_recall", 0),
            "val_precision": output.get(f"{metric_key_prefix}_precision", 0),
            "learning_rate": learning_rate,
            "patience_counter": patience_counter
        })

        return output

# 5. Evaluation Metrics --------------------------------------------
metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate standard micro accuracy (overall accuracy)
    micro_accuracy = {"micro_accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}
    
    # Calculate macro accuracy (average of per-class accuracies)
    from sklearn.metrics import balanced_accuracy_score
    macro_accuracy = {"macro_accuracy": balanced_accuracy_score(labels, predictions)}
    
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    # Add recall calculation
    from sklearn.metrics import recall_score, precision_score
    recall = {"recall": recall_score(labels, predictions, average="macro")}
    precision = {"precision": precision_score(labels, predictions, average="macro")}

    return {**micro_accuracy, **macro_accuracy, **f1, **recall, **precision}  # Merge all dictionaries to return all metrics

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
    
    return dataset

# Load tokenized datasets with the appropriate path based on the balanced flag
tokenized_dir = f"{args.dataset_name}/{model_abbreviation}/{'balanced_' if args.balanced else ''}tokenized_splits"
train_dataset_path = f"{tokenized_dir}/train_split.csv"
val_dataset_path = f"{tokenized_dir}/validation_split.csv"
test_dataset_path = f"{tokenized_dir}/test_split.csv"

train_dataset = load_tokenized_dataset(train_dataset_path)
val_dataset = load_tokenized_dataset(val_dataset_path)
test_dataset = load_tokenized_dataset(test_dataset_path)

# Create PyTorch datasets
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset objects
train_dataset_obj = TokenizedDataset(
    {'input_ids': train_dataset['input_ids'], 'attention_mask': train_dataset['attention_mask']},
    train_dataset['labels']
)
val_dataset_obj = TokenizedDataset(
    {'input_ids': val_dataset['input_ids'], 'attention_mask': val_dataset['attention_mask']},
    val_dataset['labels']
)
test_dataset_obj = TokenizedDataset(
    {'input_ids': test_dataset['input_ids'], 'attention_mask': test_dataset['attention_mask']},
    test_dataset['labels']
)

print("Successfully loaded tokenized datasets")

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_obj,
    eval_dataset=val_dataset_obj,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=50)],  # Early stopping
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

# Start Training
trainer.train()

# 7. Final Evaluation ----------------------------------------------
print("\nFinal test evaluation:")
test_results = trainer.evaluate(test_dataset_obj, metric_key_prefix="test")
print("Available keys in test_results:", test_results.keys())  # Debug: print all available keys

# Print all test results 
print("Test results:", test_results)

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

# Log all relevant metrics to wandb at the end of training
# Use dictionary.get() with default values to avoid KeyError
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, precision_score

# Get predictions for test dataset
test_predictions = trainer.predict(test_dataset_obj)
test_pred_labels = np.argmax(test_predictions.predictions, axis=1)
test_true_labels = [item['labels'].item() for item in [test_dataset_obj[i] for i in range(len(test_dataset_obj))]]

# Log all relevant metrics to wandb at the end of training
wandb.log({
    "test_micro_accuracy": test_results['test_micro_accuracy'],  # This is micro accuracy
    "test_macro_accuracy": test_results.get('test_macro_accuracy', 
                                           balanced_accuracy_score(test_true_labels, test_pred_labels)),
    "test_loss": test_results['test_loss'],
    "test_macro_f1": test_results['test_f1'],  # This is already macro F1
    "test_micro_f1": f1_score(test_true_labels, test_pred_labels, average='micro'),
    "test_macro_recall": test_results['test_recall'],  # This is macro recall
    "test_micro_recall": recall_score(test_true_labels, test_pred_labels, average='micro'),
    "test_macro_precision": test_results['test_precision'],  # This is macro precision
    "test_micro_precision": precision_score(test_true_labels, test_pred_labels, average='micro'),
})

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

# Evaluation script without using HuggingFace Trainer
print("\nEvaluating model without Trainer:")

def evaluate_model_without_trainer(model, dataset, device, split_name, save_dir):
    model.eval()
    all_predictions = []
    all_labels = []
    all_example_ids = []  # To track example IDs if available
    
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

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                logits = outputs['logits']

                # Get prediction
                prediction = torch.argmax(logits, dim=-1).item()

                # Store prediction and label
                all_predictions.append(prediction)
                all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
                all_example_ids.append(i)  # Use index as example ID

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Micro accuracy (standard overall accuracy)
    micro_accuracy = accuracy_score(all_labels, all_predictions)
    
    # Macro accuracy (balanced accuracy - average of per-class accuracies)
    macro_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    
    # Calculate macro and micro metrics
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    macro_recall = recall_score(all_labels, all_predictions, average='macro')
    macro_precision = precision_score(all_labels, all_predictions, average='macro')
    
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    micro_recall = recall_score(all_labels, all_predictions, average='micro')
    micro_precision = precision_score(all_labels, all_predictions, average='micro')
    
    # Per-class metrics
    per_class_precision = precision_score(all_labels, all_predictions, average=None)
    per_class_recall = recall_score(all_labels, all_predictions, average=None)
    per_class_f1 = f1_score(all_labels, all_predictions, average=None)
    
    # Get unique classes
    unique_classes = sorted(set(all_labels))
    
    # Create per-class metrics dictionary
    per_class_metrics = {}
    for i, class_idx in enumerate(unique_classes):
        per_class_metrics[f"class_{class_idx}"] = {
            "precision": per_class_precision[i] if i < len(per_class_precision) else 0,
            "recall": per_class_recall[i] if i < len(per_class_recall) else 0,
            "f1": per_class_f1[i] if i < len(per_class_f1) else 0
        }
    
    # Generate normalized confusion matrix (normalize by row/true labels)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Replace NaN with 0
    
    # Plot and save the normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Normalized Confusion Matrix - {split_name}')
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the confusion matrix plot
    confusion_matrix_path = os.path.join(save_dir, f'confusion_matrix_{split_name}.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Get detailed classification report as string
    class_report = classification_report(all_labels, all_predictions, output_dict=True)

    # Save predictions to CSV
    import pandas as pd
    predictions_df = pd.DataFrame({
        'example_id': all_example_ids,
        'true_label': all_labels,
        'predicted_label': all_predictions
    })
    predictions_path = os.path.join(save_dir, f'predictions_{split_name}.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    return {
        'micro_accuracy': micro_accuracy,
        'macro_accuracy': macro_accuracy,
        'macro_f1': macro_f1,
        'macro_recall': macro_recall,
        'macro_precision': macro_precision,
        'micro_f1': micro_f1,
        'micro_recall': micro_recall,
        'micro_precision': micro_precision,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix_norm': conf_matrix_norm.tolist(),
        'confusion_matrix_path': confusion_matrix_path,
        'classification_report': class_report,
        'predictions_path': predictions_path,  # Add path to saved predictions
        'predictions': {  # Also include predictions in the results
            'example_ids': all_example_ids,
            'true_labels': all_labels,
            'predicted_labels': all_predictions
        }
    }

# Use the already loaded tokenized datasets
validation_dataset = val_dataset_obj
test_dataset = test_dataset_obj
print("Successfully loaded and tokenized datasets")

# Load the model
model = Classifier(args.model_name, args.dataset_name)
state_dict = load_file(f"{model_path}/model.safetensors")
model.load_state_dict(state_dict)
model.to(device)

# Make sure the model path directory exists
os.makedirs(model_path, exist_ok=True)

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

# After evaluating on validation and test sets
print(f"\nPredictions saved to:")
print(f"  Validation: {val_metrics['predictions_path']}")
print(f"  Test: {test_metrics['predictions_path']}")

# Save results
import json
results = {
    "validation": {k: v for k, v in val_metrics.items() if k != 'classification_report'},
    "test": {k: v for k, v in test_metrics.items() if k != 'classification_report'}
}

# Add classification report separately (it's a nested dict)
results["validation"]["classification_report"] = val_metrics["classification_report"]
results["test"]["classification_report"] = test_metrics["classification_report"]

# Convert numpy arrays to lists for JSON serialization
if isinstance(results["validation"]["confusion_matrix_norm"], np.ndarray):
    results["validation"]["confusion_matrix_norm"] = results["validation"]["confusion_matrix_norm"].tolist()
if isinstance(results["test"]["confusion_matrix_norm"], np.ndarray):
    results["test"]["confusion_matrix_norm"] = results["test"]["confusion_matrix_norm"].tolist()

with open(f"{model_path}/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {model_path}/evaluation_results.json")