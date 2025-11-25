# This code is for the pretext compression pretraining stage of LLM2Comp
import json
from transformers import LlamaTokenizer, AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
from models import (
    Llm2Comp_NLL_First_Stage_Bidirection,
    Llm2Comp_KL_First_Stage_Bidirection,
    Llm2Comp_Rc_First_Stage_Bidirection,
    Llm2Comp_NLL_First_Stage_Bidirection_Checkpoint,
    Llm2Comp_KL_First_Stage_Bidirection_Checkpoint,
    Llm2Comp_Rc_First_Stage_Bidirection_Checkpoint,
)
from transformers import AdamW, HfArgumentParser, get_scheduler
from arguments_first_stage import ModelArguments, DataTrainingArguments, TrainingArguments
import torch
from tqdm import tqdm
import logging
import os
import deepspeed
import wandb
import copy
from itertools import chain
import random

# Parse command-line arguments into dataclass objects
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Load the tokenizer (authentication token removed for security)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name
)

# Configure tokenizer based on the model type
if "llama" in model_args.model_name.lower():
    tokenizer.pad_token = tokenizer.eos_token
elif "mistral" in model_args.model_name.lower():
    tokenizer.mask_token = "_"
    tokenizer.pad_token_id = 583
    print("------------------------Mistral-----------------------------")


def load_wiki_dataset(data_args, training_args):
    """Load and preprocess the WikiText-103 dataset for continuation training."""

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset["train"]

    add_special_token_id = data_args.special_token_id
    add_sequence_length = data_args.sequence_length
    max_length = data_args.max_length

    def tokenize(examples):
        encoded_inputs = []
        continuation_labels = []
        whole_input_ids = []
        continuation_length = []

        for text in examples["text"]:
            input_ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")[
                "input_ids"
            ][0]
            whole_input_ids.append(input_ids)

            # Choose continuation site randomly, ensuring at least 10 tokens remain
            continuation_site = random.randint(
                10, len(input_ids) - 10
            )

            encoded_inputs.append(input_ids[:continuation_site])
            continuation_length.append(len(input_ids[continuation_site:]))
            continuation_labels.append(
                torch.cat([input_ids[:1], input_ids[continuation_site:]])
            )

        # Insert special token IDs at the end of each sequence
        inserted_input_ids = []
        site_mem = []
        for input_id in encoded_inputs:
            inserted_sequence = torch.cat(
                [
                    input_id,
                    torch.tensor(
                        list(
                            range(
                                add_special_token_id,
                                add_special_token_id + add_sequence_length,
                            )
                        )
                    ),
                ]
            )

            if len(inserted_sequence) > max_length:
                site_mem.append(max_length - add_sequence_length)
                inserted_sequence = torch.cat(
                    [
                        inserted_sequence[: max_length - add_sequence_length],
                        torch.tensor(
                            list(
                                range(
                                    add_special_token_id,
                                    add_special_token_id + add_sequence_length,
                                )
                            )
                        ),
                    ]
                )
            else:
                site_mem.append(len(input_id))
            inserted_input_ids.append(inserted_sequence)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            inserted_input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        padded_whole_input_ids = torch.nn.utils.rnn.pad_sequence(
            whole_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )

        embedding_indices = []
        for input_id_index in range(len(padded_input_ids)):
            site = site_mem[input_id_index]
            embedding_indices.append([input_id_index, site])

        padded_continuation_labels = torch.nn.utils.rnn.pad_sequence(
            continuation_labels,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        return {
            "input_ids": padded_input_ids,
            "embedding_indices": torch.tensor(embedding_indices),
            "continuation_labels": padded_continuation_labels,
            "whole_input_ids": padded_whole_input_ids,
            "continuation_length": torch.tensor(continuation_length),
        }

    def is_not_empty(example):
        return example["text"] is not None and example["text"] != ""

    def filter_long_sentences(example):
        return len(example["text"]) >= 1000

    def filter_suitable_sentences(example):
        return (
            len(
                tokenizer(
                    example["text"], add_special_tokens=True, return_tensors="pt"
                )["input_ids"][0]
            )
            < 512
        )

    filtered_dataset = dataset.filter(is_not_empty)
    filtered_dataset = filtered_dataset.filter(filter_long_sentences)
    filtered_dataset = filtered_dataset.filter(filter_suitable_sentences)
    filtered_dataset = filtered_dataset.select(range(128000))

    encode_ds = filtered_dataset.map(
        tokenize,
        batched=True,
        batch_size=training_args.train_batch_size,
        drop_last_batch=True,
    )
    encode_ds.set_format(
        type="torch",
        columns=[
            "input_ids",
            "embedding_indices",
            "continuation_labels",
            "whole_input_ids",
            "continuation_length",
        ],
    )

    dataloader = DataLoader(
        encode_ds, batch_size=training_args.train_batch_size
    )
    return dataloader


def load_wiki_dataset_for_reconstruction(data_args, training_args):
    """Load and preprocess the WikiText-103 dataset for reconstruction tasks."""

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset["train"]

    add_special_token_id = data_args.special_token_id
    add_sequence_length = data_args.sequence_length
    max_length = data_args.max_length

    def tokenize(examples):
        whole_input_ids = []
        whole_length = []
        for text in examples["text"]:
            input_ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")[
                "input_ids"
            ][0]
            whole_input_ids.append(input_ids)
            whole_length.append(len(input_ids))

        inserted_input_ids = []
        site_mem = []
        for input_id in whole_input_ids:
            inserted_sequence = torch.cat(
                [
                    input_id,
                    torch.tensor(
                        list(
                            range(
                                add_special_token_id,
                                add_special_token_id + add_sequence_length,
                            )
                        )
                    ),
                ]
            )

            if len(inserted_sequence) > max_length:
                site_mem.append(max_length - add_sequence_length)
                inserted_sequence = torch.cat(
                    [
                        inserted_sequence[: max_length - add_sequence_length],
                        torch.tensor(
                            list(
                                range(
                                    add_special_token_id,
                                    add_special_token_id + add_sequence_length,
                                )
                            )
                        ),
                    ]
                )
            else:
                site_mem.append(len(input_id))
            inserted_input_ids.append(inserted_sequence)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            inserted_input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        embedding_indices = []
        for input_id_index in range(len(padded_input_ids)):
            site = site_mem[input_id_index]
            embedding_indices.append([input_id_index, site])

        return {
            "input_ids": padded_input_ids,
            "embedding_indices": torch.tensor(embedding_indices),
        }

    def is_not_empty(example):
        return example["text"] is not None and example["text"] != ""

    def filter_long_sentences(example):
        return len(example["text"]) >= 1000

    def filter_suitable_sentences(example):
        return (
            len(
                tokenizer(
                    example["text"], add_special_tokens=True, return_tensors="pt"
                )["input_ids"][0]
            )
            < 512
        )

    filtered_dataset = dataset.filter(is_not_empty)
    filtered_dataset = filtered_dataset.filter(filter_long_sentences)
    filtered_dataset = filtered_dataset.filter(filter_suitable_sentences)
    filtered_dataset = filtered_dataset.select(range(32000))

    encode_ds = filtered_dataset.map(
        tokenize,
        batched=True,
        batch_size=training_args.train_batch_size,
        drop_last_batch=True,
    )
    encode_ds.set_format(type="torch", columns=["input_ids", "embedding_indices"])
    return encode_ds


# Configure task type and corresponding model/dataset class
task_type = training_args.task_type
save_path = (
    f"{task_type}-pretext-compression-Bidirectional-llama-2-"
    f"{data_args.sequence_length}tokens-128000samples"
)
model_class = None
dataset_class = None
log_file = None

if task_type == "NLL":
    model_class = Llm2Comp_NLL_First_Stage_Bidirection
    dataset_class = load_wiki_dataset
    log_file = f"wiki_nll_loss_{data_args.sequence_length}tokens"
elif task_type == "KL":
    model_class = Llm2Comp_KL_First_Stage_Bidirection
    dataset_class = load_wiki_dataset
    log_file = f"wiki_kl_loss_{data_args.sequence_length}tokens"
elif task_type == "Rc":
    model_class = Llm2Comp_Rc_First_Stage_Bidirection_Checkpoint
    dataset_class = load_wiki_dataset_for_reconstruction
    log_file = f"wiki_rc_loss_{data_args.sequence_length}tokens"
else:
    print(f"Error Task Type: {task_type}")

# Logger configuration
logger = logging.getLogger(name="my_logger")
os.makedirs("./ablation_study", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("./ablation_study", f"{log_file}.log"),
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)

# Load dataset and initialize model
dataLoader = dataset_class(data_args, training_args)
model = model_class(data_args, training_args.local_rank)

# Print names and shapes of trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter Name: {name}\tParameter Shape: {param.size()}")

# Initialize DeepSpeed engine
model_engine, _, _, _ = deepspeed.initialize(
    args=training_args,
    config_params=training_args.deepspeed,
    model=model,
    model_parameters=model.parameters(),
)

torch.cuda.set_device(training_args.local_rank)

# Training loop
for epoch in range(training_args.train_epoch):
    model_engine.train()
    for idx, batch in enumerate(tqdm(dataLoader, desc=f"Epoch: {epoch+1}")):
        batch = {k: i.cuda() for k, i in batch.items()}
        loss = model_engine(**batch)
        model_engine.backward(loss)
        if training_args.local_rank == 0:
            logger.info(f"Epoch: {epoch+1}, Batch:{idx+1}, Loss: {loss}")
        model_engine.step()

    # Save checkpoint after each epoch (path generalized)
    model_engine.save_checkpoint("./checkpoints", f"{save_path}")
    del model_engine
