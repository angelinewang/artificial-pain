# Toward Affective Empathy in AI: Encoding Internal Representations of Artificial Pain

### Angeline Wang and Iran R. Roman
#### Department of Computer Science, Queen Mary University of London 

## Running our code 
Install all necessary packages:
```
pip install -r requirements.txt
```

Set these environment variables for HuggingFace access and Weights and Biases tracking:
```
import os

# Set the Hugging Face token as an environment variable
os.environ['HF_TOKEN'] = ''

# Verify that the token is set (optional)
print(os.environ.get('HF_TOKEN'))

os.environ['WANDB_API_KEY']= ''
```


### **Classification-only** model is in the 'classification' folder---functional for both SAD and ESConv datasets. 

First script in each folder is for loading data, and second script in each folder is for training and evaluating the model. 

#### 1. Run classification-dataloader.py with HuggingFace Token and relevant flags
```
python classification-dataloader.py --model_name answerdotai/ModernBERT-base --dataset_name ESConv --balanced --samples_per_label 1500
```
Flag options:

'--model_name': default="mental/mental-roberta-base", choices=["mental/mental-roberta-base", "mental/mental-bert-base-uncased", "answerdotai/ModernBERT-base"],

'--dataset_name': default="ESConv", choices=["ESConv", "SAD"],

'--balanced': action='store_true'

Run with '--balanced' flag if you want to balance the dataset to have equal samples per label. 
Without this flag, the dataset created will retain the original distribution. 

'--samples_per_label': default=1500

This sets the number of samples per label when balancing. 

#### 2. Run classification-model.py with Hugging Face Token and Weights and Biases API Key
```
python classification-model.py --model_name mental/mental-bert-base-uncased --dataset_name ESConv --balanced
```
Flag options:

'--model_name': default="mental/mental-roberta-base", choices=["mental/mental-roberta-base", "mental/mental-bert-base-uncased", "answerdotai/ModernBERT-base"]

'--dataset_name': default="ESConv", choices=["ESConv", "SAD"]

'--debug', action='store_true'

Run with '--debug' if you want to enable debug mode, which only trains the model for 3 epochs. 

'--balanced': action='store_true'

Run with '--balanced' to use the created balanced dataset for training and evaluation. 


### **Multi-task model** is in the 'intensity' folder---only functional for the ESConv dataset. 

First script in each folder is for loading data, and second script in each folder is for training and evaluating the model. 

#### 1. Run intensity-dataloader.py with HuggingFace Token and relevant flags
```
python intensity-dataloader.py --model_name mental/mental-roberta-base --dataset_name ESConv
```
Flag options:

'--model_name': default="mental/mental-roberta-base", choices=["mental/mental-roberta-base", "mental/mental-bert-base-uncased", "answerdotai/ModernBERT-base"]

'--dataset_name': default="ESConv", choices=["ESConv"]

ESConv is the only dataset available for the intensity multi-task model dataset creation. 

'--balanced': action='store_true'

Run with '--balanced' flag if you want to balance the dataset to have equal samples per label. 
Without this flag, the dataset created will retain the original distribution. 

'--samples_per_label',: default=1500

This sets the number of samples per label when balancing. 

#### 2. Run intensity-model.py with Hugging Face Token and Weights and Biases API Key
```
python intensity-model.py --model_name answerdotai/ModernBERT-base --dataset_name ESConv --head_num 4 --model_metric accuracy
```
Flag options:

'--model_name': default="mental/mental-roberta-base", choices=["mental/mental-roberta-base", "mental/mental-bert-base-uncased", "answerdotai/ModernBERT-base"]

'--dataset_name': default="ESConv", choices=["ESConv"]

ESConv is the only dataset available for the intensity multi-task model. 

'--debug', action='store_true'

Run with '--debug' if you want to enable debug mode, which only trains the model for 3 epochs. 

'--balanced': action='store_true'

Run with '--balanced' to use the created balanced dataset for training and evaluation. 

'--head_num': default=4

Do not change this setting, as the script only runs for the model with 4 heads. 

'--model_metric': default="accuracy"

Choose the metric for model evaluation, either accuracy or loss is most suitable. Accuracy worked better for us. 

## Contact 
If you have any questions about this work, please contact *Angeline Wang* at [ec24817@qmul.ac.uk](mailto:ec24817@qmul.ac.uk).

## Reference 
If you use our dataset or code, please cite our paper: [Toward Affective Empathy in AI: Encoding Internal Representations of Artificial Pain
](). 
```
@article{wangroman2025artificialpain,
    title = {Toward Affective Empathy in AI: Encoding Internal Representations of Artificial Pain},
    author={Wang, Angeline and Roman, Iran R},
    year = {2025},
    publisher = {8th annual conference on Cognitive Computational Neuroscience},
    keywords = {empathy; emotional support; multi-task model;
    brain-inspired ai; affective computing; representational learning},
  }
```