import json
from transformers import LlamaTokenizer, AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader, DistributedSampler
from models import Llm2Comp_Second_Stage_Bidirection
from transformers import (
    AdamW,
    HfArgumentParser,
    get_scheduler,
)

from arguments_second_stage import ModelArguments, DataTrainingArguments, TrainingArguments
import torch
from tqdm import tqdm
import logging
import os
import deepspeed
import wandb
import copy
from itertools import chain
import numpy as np
import random
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset, load_from_disk

# Parse command line arguments
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
if 'llama' in model_args.model_name.lower(): 
    tokenizer.pad_token = tokenizer.eos_token
elif 'mistral' in model_args.model_name.lower():
    tokenizer.mask_token = "_"
    tokenizer.pad_token_id = 583
    print('------------------------Mistral tokenizer initialized-----------------------------')

# Initialize WandB
wandb.init(project="wiki_simcse", name="wiki_simcse")


def load_data(file_path: str = None):
    """Load plain text data from a file (one line per sample)."""
    all_data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            all_data.append(line)
    return all_data
                

def set_seed(seed):
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure determinism in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def build_dataset(data_args, training_args):
    """Build dataset for training with inserted special tokens."""
    set_seed(42)
    data_pieces = load_data(data_args.path)
    random.shuffle(data_pieces)
    dataset = Dataset.from_dict({'text': data_pieces[:128000]})

    add_special_token_id = data_args.special_token_id
    add_sequence_length = data_args.sequence_length
    max_length = data_args.max_length

    def is_not_empty(example):
        return example['text'] is not None and example['text'] != ''

    def tokenize(examples):
        encoded_inputs = [tokenizer(text, add_special_tokens=True, return_tensors='pt')['input_ids'][0]
                          for text in examples['text']]
        inserted_input_ids, site_mem = [], []
        for input_id in encoded_inputs:
            inserted_sequence = torch.cat([
                input_id,
                torch.tensor(list(range(add_special_token_id, add_special_token_id + add_sequence_length)))
            ])
            # Truncate if exceeding max_length while keeping inserted tokens at the end
            if len(inserted_sequence) > max_length:
                site_mem.append(max_length - add_sequence_length)
                inserted_sequence = torch.cat([
                    inserted_sequence[:max_length - add_sequence_length],
                    torch.tensor(list(range(add_special_token_id, add_special_token_id + add_sequence_length)))
                ])
            else:
                site_mem.append(len(input_id))
            inserted_input_ids.append(inserted_sequence)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            inserted_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        embedding_indices = [[i, site_mem[i]] for i in range(len(padded_input_ids))]

        return {
            'input_ids': padded_input_ids,
            'embedding_indices': torch.tensor(embedding_indices)
        }

    filtered_dataset = dataset.filter(is_not_empty)
    encode_ds = filtered_dataset.map(tokenize, batched=True, batch_size=training_args.train_batch_size, drop_last_batch=True)
    encode_ds.set_format(type='torch', columns=['input_ids', 'embedding_indices'])

    # Do not shuffle here because padding has been applied during map
    dataloader = DataLoader(encode_ds, batch_size=training_args.train_batch_size)
    return dataloader



def get_distributed_dataloader(dataset, batch_size, shuffle=False):
    """Create a distributed DataLoader for multi-GPU training."""
    sampler = DistributedSampler(dataset, num_replicas=torch.distributed.get_world_size(),
                                 rank=torch.distributed.get_rank(), shuffle=shuffle, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# Set up logging
logger = logging.getLogger(name='my_logger')
os.makedirs('./wiki_train', exist_ok=True)
logging.basicConfig(filename=os.path.join('./wiki_train', 'second_stage.log'),
                    level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')


model_class = Llm2Comp_Second_Stage_Bidirection
dataset_class = build_dataset

# Prepare dataloader and model
dataLoader = dataset_class(data_args, training_args)
model = model_class(data_args, training_args.local_rank)

checkpoint = torch.load(model_args.checkpoint_path)
model.load_state_dict(checkpoint['module'], strict=True)

# Free memory from the original model if not needed
import gc
del model.origin_model
gc.collect()
torch.cuda.empty_cache()

# Merge LoRA weights
model.model.merge_and_unload()
lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)
model.model = get_peft_model(model.model, lora_config)
model.set_attn_dropout(training_args.dropout_rate)
model.model_wrapper()

# Show trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter: {name}\tShape: {param.size()}")

# Initialize DeepSpeed engine
model_engine, _, _, _ = deepspeed.initialize(
    args=training_args,
    config_params=training_args.deepspeed,
    model=model,
    model_parameters=model.parameters()
)

# Training loop
torch.cuda.set_device(training_args.local_rank)
for epoch in range(training_args.train_epoch):
    model_engine.train()
    for idx, batch in enumerate(tqdm(dataLoader, desc=f'Epoch: {epoch+1}')):
        batch = {k: v.cuda() for k, v in batch.items()}
        loss = model_engine(**batch)
        model_engine.backward(loss)
        if training_args.local_rank == 0:
            logger.info(f'Epoch: {epoch+1}, Batch: {idx+1}, Loss: {loss}')
            wandb.log({"loss": loss})
        model_engine.step()

    # Save model checkpoint
    model_engine.save_checkpoint(training_args.save_dir, f'{model_args.save_dir}')
    del model_engine
