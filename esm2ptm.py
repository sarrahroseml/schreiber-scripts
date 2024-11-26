import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle 
import os
import wandb
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from datasets import Dataset
from accelerate import Accelerator
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

def get_ptm_sites(row):
    #Extract pos of modified residues from modified residue col
    modified_positions = [int(i) for i in re.findall(r'MOD_RES (\d+)', row['Modified residue'])]
    #create list of 0s of len equal to protein seq 
    ptm_sites = [0] * len(row['Sequence'])

    #replace zeros w ones at pos of modified residues 
    for pos in modified_positions:
        #Subtract 1 bec pos are 1-indexed but lists are 0 indexed
        ptm_sites[position - 1] = 1
    return ptm_sites 

#applyy fn to each row in df
data['PTM sites'] = data.apply(get_ptm_sites, axis=1)

def split_into_chunks(row):
    """"Split longer protein  sequences & their labes into non-overlapping chunks of <=512 bec of smaller context window of 1024""""
    seq = row['Sequence']
    ptm_sites = row['PTM sites']
    chunk_size = 512

    num_chunks = (len(sequence) + chunk_size - 1) // chunk_size  
    #Split sequences & PTM sites into chunks 
    sequence_chunks = [sequence[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    ptm_sites_chunks = [ptm_sites[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    
    #Create new rows for each chunk 
    rows = []
    for i in range(num_chunks):
        new_row = row.copy()
        new_row['Sequence'] = sequence_chunks[i]
        new_row['PTM sites'] = ptm_sites_chunks[i]
        rows.append(new_row)
    
    return rows

#create a new df to store chunks 
chunks_data = []

#Iterate through each row of original df and split into chunks 
for _, row in data.iterrows():
    chunks_data.extend(split_into_chunks(row))

#Convert list of chunks into a df
chunks_df = pd.DataFrame(chunks_data)

#Reset index of df
chunks_df.reset_index(drop=True, inplace=True)


#Split data into train and test based on families
def split_data(df): 
    #Get unique list of protein families 
    unique_families = df['Protein families'].unique().tolist()
    np.random.shuffle(unique_families)
        test_data = []
    test_families = []
    total_entries = len(df)
    total_families = len(unique_families)

    with tqdm(total=total_families) as pbar:
        for family in unique_families:
            #Separate all proteins in current famility into test data 
            family_data = df[df['Protein families'] == family]
            test_data.append(family_data) 
            test_families.append(family) 
            #Remove curr family data from original df
            df = df[df['Protein families'] != family]

            #Calc %of test data & %of families in test data
            percent_test_data = sum(len(data) for data in test_data) / total_entries * 100
            percent_test_families = len(test_families) / total_families * 100

            #Update tqdm progress bar 
            pbar.set_description(f'% Test Data: {percent_test_data:.2f}% | % Test Families: {percent_test_families:.2f}%')
            pbar.update(1) 
            
            #Check if 20% threshold for test data is crossed 
            if percent_test_data >= 20: 
                break 

    #Concatenate list of test data dfs into a single df 
    test_data = pd.concat(test_data, ignore_index = True)

    return df, test_df 

train_df, test_df = split_data(chunks_df) 

#Reduct size of datasets
fraction = 1.00
reduced_train_df = train_df.sample(frac=fraction, random_state=42)
reduced_test_df = test_df.sample(frac=fraction, random_state=42)

#Extract sequences & ptm site labels
train_sequences_reduced = reduced_train_df['Sequence'].tolist()
train_labels_reduced = reduced_train_df['PTM sites'].tolist()
test_sequences_reduced = reduced_test_df['Sequence'].tolist()
test_labels_reduced = reduced_test_df['PTM sites'].tolist()

pickle_file_path = "2100K_ptm_data_512/"

with open(pickle_file_path + "train_sequences_chunked_by_family.pkl", "wb") as f:
    pickle.dump(train_sequences_reduced, f)

with open(pickle_file_path + "test_sequences_chunked_by_family.pkl", "wb") as f:
    pickle.dump(test_sequences_reduced, f)

with open(pickle_file_path + "train_labels_chunked_by_family.pkl", "wb") as f:
    pickle.dump(train_labels_reduced, f)

with open(pickle_file_path + "test_labels_chunked_by_family.pkl", "wb") as f:
    pickle.dump(test_labels_reduced, f)


#Return the paths to saved pixkle files 
saved_files = [
    pickle_file_path + "train_sequences_chunked_by_family.pkl",
    pickle_file_path + "test_sequences_chunked_by_family.pkl",
    pickle_file_path + "train_labels_chunked_by_family.pkl",
    pickle_file_path + "test_labels_chunked_by_family.pkl"
]


def print_trainable_parameters(model): 
    """ Print num of trainable params"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}") 

def save_config_to_txt(config, filename): 
    """Save config dictionary to a txt file"""
    with open(filename, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def truncate_labels(labels, max_length): 
    return [label[:max_length] for label in labels]

def compute_metrics(p): 
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions[labels != -100].flatten()
    labels = labels[labels != -100].flatten()
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc = roc_auc_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mcc': mcc}

def compute_loss(model, logits, inputs): 
    labels = inputs["labels"]
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    active_loss = inputs["attention_mask"].view(-1) == 1
    active_logits = logits.view(-1, model.config.num_labels)
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )
    loss = loss_fct(active_logits, active_labels)
    return loss

#Load data from pickle files 
with open("2100K_ptm_data/train_sequences_chunked_by_family.pkl", "rb") as f:
    train_sequences = pickle.load(f)
    
with open("2100K_ptm_data/test_sequences_chunked_by_family.pkl", "rb") as f:
    test_sequences = pickle.load(f)

with open("2100K_ptm_data/train_labels_chunked_by_family.pkl", "rb") as f:
    train_labels = pickle.load(f)

with open("2100K_ptm_data/test_labels_chunked_by_family.pkl", "rb") as f:
    test_labels = pickle.load(f)

#Tokenisation 
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

#Set max seq length to the tokenizer's max input length 
max_sequence_length = 1024

train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False, add_special_tokens=False)
test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False, add_special_tokens=False)

#truncate list of labels 
train_labels = truncate_labels(train_labels, max_sequence_length)
test_labels = truncate_labels(test_labels, max_sequence_length)

#Create datasets 
train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)

#Compute class weights 
classes = [0, 1]
flat_train_labels = [label for sublist in train_labels for label in sublist]
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=flat_train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(accelerator.device)

#Define custom trainer class
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        loss = compute_loss(model, logits, inputs)
        return (loss, outputs) if return_outputs else loss

#Configure quantisation settings 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def train_function_no_sweeps(train_datasets, test_dataset): 
    """Set model config, init ems2 model, prepare for peft & quantisation training, training args, etc."""

    #Set the config
    config = {
        "lora_alpha": 1,
        "lora_dropout": 0.5,
        "lr": 3.701568055793089e-04,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.5,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 36,
        "r": 2,
        "weight_decay": 0.3,
    }

    #Log the config 
    wandb.config.update(config)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config_filename = f"esm2_t30_150M_qlora_ptm_config_{timestamp}.txt"
    save_config_to_txt(config, config_filename)

    model_checkpoint = "facebook/esm2_t30_150M_UR50D"

    #Define labels and model 
    id2label = {0: "No ptm site", 1: "ptm site"}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config  
    )

    #Prepare model into a PeftModel 
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=[
            "query",
            "key",
            "value",
            "EsmSelfOutput.dense",
            "EsmIntermediate.dense",
            "EsmOutput.dense",
            "EsmContactPredictionHead.regression",
            "classifier"
        ],
        lora_dropout=config["lora_dropout"],
        bias="none",  # or "all" or "lora_only"
        # modules_to_save=["classifier"]
    )

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    #Use accelerator 
    model = accelerator.prepare(model)
    train_dataset = accelerator.prepare(train_dataset) 
    test_dataset = accelerator.prepare(test_dataset) 

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    training_args = TrainingArguments(
        output_dir=f"esm2_t30_150M_qlora_ptm_sites_{timestamp}",
        learning_rate=config["lr"],
        lr_scheduler_type=config["lr_scheduler_type"],
        gradient_accumulation_steps=1, # changed from 1 to 4
        # warmup_steps=2, # added this in
        max_grad_norm=config["max_grad_norm"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir=None,
        logging_first_step=False,
        logging_steps=200,
        save_total_limit=3,
        no_cuda=False,
        seed=8893,
        fp16=True,
        report_to='wandb',
        optim="paged_adamw_8bit" 

    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )


    trainer.train()
    save_path = os.path.join("qlora_ptm_sites", f"best_model_esm2_t30_150M_qlora_{timestamp}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    train_function_no_sweeps(train_dataset, test_dataset)
