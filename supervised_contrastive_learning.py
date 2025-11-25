import json
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
from models import Llm2Comp_Third_Stage_Bidirection_Gather
from transformers import (
    AdamW,
    HfArgumentParser,
    get_scheduler,
)
from arguments_third_stage import ModelArguments, DataTrainingArguments, TrainingArguments
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
from loss_utils import cos_sim, mismatched_sizes_all_gather
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

# Parse training arguments
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
if 'llama' in model_args.model_name.lower(): 
    tokenizer.pad_token = tokenizer.eos_token
elif 'mistral' in  model_args.model_name.lower():
    tokenizer.mask_token = "_"
    tokenizer.pad_token_id = 583
    print('------------------------Mistral tokenizer initialized-----------------------------')

# Initialize Weights & Biases (with environment credentials)
wandb.init(project="wiki_simcse", name="wiki_simcse")

# Task-specific instructions
Instructions = {
    'allnli': 'Given a premise, retrieve a hypothesis that is entailed by the premise Retrieve semantically similar text',
    'dureader': 'Given a Chinese search query, retrieve web passages that answer the question',
    'eli5_question_answer': 'Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum',
    'fever': 'Given a claim, retrieve documents that support or refute the claim',
    'hotpot_qa': 'Given a multi-hop question, retrieve documents that can help answer the question',
    'miracl': 'Given a question, retrieve Wikipedia passages that answer the question',
    'mrtydi': 'Given a question, retrieve Wikipedia passages that answer the question',
    'msmarco_document': 'Given a web search query, retrieve relevant documents that answer the query',
    'msmarco_passage': 'Given a web search query, retrieve relevant passages that answer the query',
    'nq': 'Given a question, retrieve Wikipedia passages that answer the question',
    'quora_duplicates': [
        "Given a question, retrieve questions that are semantically equivalent to the given question",
        "Find questions that have the same meaning as the input question",
    ],
    'squad': 'Retrieve Wikipedia passages that answer the question',
    't2ranking': 'Given a Chinese search query, retrieve web passages that answer the question',
    'trivia_qa': 'Retrieve Wikipedia passages that answer the question'
}

Instruction_Prompt = "Please identify and output the most important content that answers the following question:\n"


def get_distributed_dataloader(dataset, batch_size, shuffle=False):
    """Create distributed DataLoader for multi-GPU training."""
    sampler = DistributedSampler(dataset,
                                 num_replicas=torch.distributed.get_world_size(),
                                 rank=torch.distributed.get_rank(),
                                 shuffle=shuffle,
                                 drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def load_data_and_sampling(file_path: str = None):
    """Load multiple datasets and sample a fixed number of examples."""
    all_files = os.listdir(file_path)
    all_data = []
    idx = 0
    for every_file in tqdm(all_files):
        print(every_file)
        now_file = os.path.join(file_path, every_file)
        with open(now_file, "r") as f:
            for line in f:
                idx += 1
                instruction = (
                    Instructions[every_file[:-6]]
                    if isinstance(Instructions[every_file[:-6]], str)
                    else Instructions[every_file[:-6]][idx % 2]
                )
                line = line.strip()
                a_dict = json.loads(line)
                a_dict['query'] = instruction + ':' + a_dict['query'] + '\n'
                all_data.append(a_dict)

    samples = random.sample(all_data, 1024000)  # Select 1M+ samples
    query, positive, negative = [], [], []
    for sample in samples:
        query.append(sample['query'])
        positive.append(sample['positive'])
        negative.append(sample['negative'])
    return query, positive, negative


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class MyDataset(Dataset):
    """Wrapper class for HuggingFace-style Dataset."""
    def __init__(self, data_list):
        self._data = data_list

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def build_dataset(data_args, training_args):
    """Tokenize and preprocess dataset for third-stage training."""
    set_seed(2025)
    query, positive, negative = load_data_and_sampling(data_args.path)
    dataset = Dataset.from_dict({'query': query, 'positive': positive, 'negative': negative})
    dataset = dataset.shuffle(seed=42)

    add_special_token_id = data_args.special_token_id
    add_sequence_length = data_args.sequence_length
    max_length = data_args.max_length

    def tokenize(examples):
        # ---------------------------
        # Process query texts
        # ---------------------------
        encoded_queries = [
            tokenizer(text, add_special_tokens=True, return_tensors='pt')['input_ids'][0]
            for text in examples['query']
        ]
        query_inserted = []
        query_site_mem = []
        for input_id in encoded_queries:
            # Append special token IDs
            inserted_sequence = torch.cat([
                input_id,
                torch.tensor(list(range(add_special_token_id, add_special_token_id + add_sequence_length)))
            ])
            if len(inserted_sequence) > max_length:
                query_site_mem.append(max_length - add_sequence_length)
                inserted_sequence = torch.cat([
                    inserted_sequence[:max_length - add_sequence_length],
                    torch.tensor(list(range(add_special_token_id, add_special_token_id + add_sequence_length)))
                ])
            else:
                query_site_mem.append(len(input_id))
            query_inserted.append(inserted_sequence)

        query_padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            query_inserted, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        query_embedding_indices = [
            [i, query_site_mem[i]] for i in range(len(query_padded_input_ids))
        ]

        # ---------------------------
        # Process positive texts
        # ---------------------------
        encoded_pos = [
            tokenizer(text, add_special_tokens=True, return_tensors='pt')['input_ids'][0]
            for text in examples['positive']
        ]
        pos_inserted, pos_site_mem, continuation_input_ids = [], [], []
        for input_id in encoded_pos:
            inserted_sequence = torch.cat([
                input_id,
                torch.tensor(list(range(add_special_token_id, add_special_token_id + add_sequence_length)))
            ])
            if len(inserted_sequence) > 2 * max_length:
                pos_site_mem.append(2 * max_length - add_sequence_length)
                inserted_sequence = torch.cat([
                    inserted_sequence[:2 * max_length - add_sequence_length],
                    torch.tensor(list(range(add_special_token_id, add_special_token_id + add_sequence_length)))
                ])
            else:
                pos_site_mem.append(len(input_id))
            continuation_input_ids.append(input_id[:2 * max_length] if len(input_id) > 2 * max_length else input_id)
            pos_inserted.append(inserted_sequence)

        positive_padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            pos_inserted, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        continual_padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            continuation_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        positive_embedding_indices = [
            [i, pos_site_mem[i]] for i in range(len(positive_padded_input_ids))
        ]

        # ---------------------------
        # Process negative texts
        # ---------------------------
        encoded_neg = [
            tokenizer(text, add_special_tokens=True, return_tensors='pt')['input_ids'][0]
            for text in examples['negative']
        ]
        neg_inserted, neg_site_mem = [], []
        for input_id in encoded_neg:
            inserted_sequence = torch.cat([
                input_id,
                torch.tensor(list(range(add_special_token_id, add_special_token_id + add_sequence_length)))
            ])
            if len(inserted_sequence) > 2 * max_length:
                neg_site_mem.append(2 * max_length - add_sequence_length)
                inserted_sequence = torch.cat([
                    inserted_sequence[:2 * max_length - add_sequence_length],
                    torch.tensor(list(range(add_special_token_id, add_special_token_id + add_sequence_length)))
                ])
            else:
                neg_site_mem.append(len(input_id))
            neg_inserted.append(inserted_sequence)

        negative_padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            neg_inserted, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        negative_embedding_indices = [
            [i, neg_site_mem[i]] for i in range(len(negative_padded_input_ids))
        ]

        # ---------------------------
        # Create QA mixed samples for continuation labels
        # ---------------------------
        qa_mix_examples = []
        continuation_length = []
        cont_input_ids = []
        for idx in range(len(examples['query'])):
            qa_mix_examples.append(examples['query'][idx] + examples['positive'][idx])
            continuation_length.append(positive_embedding_indices[idx][1])

        encoded_mix = [
            tokenizer(text, add_special_tokens=True, return_tensors='pt')['input_ids'][0]
            for text in qa_mix_examples
        ]
        inserted_mix = []
        for idx, seq in enumerate(encoded_mix):
            if len(seq) > 2 * max_length:
                seq = seq[:2 * max_length]
            inserted_mix.append(seq)
            begin_ptr = query_embedding_indices[idx][1]
            end_ptr = begin_ptr + continuation_length[idx]
            now_continuation_input_ids = torch.cat(
                [torch.tensor([1]), seq[begin_ptr - 1:end_ptr]], dim=0
            )
            cont_input_ids.append(now_continuation_input_ids)

        whole_input_ids = torch.nn.utils.rnn.pad_sequence(
            inserted_mix, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        padded_continuation_labels = torch.nn.utils.rnn.pad_sequence(
            cont_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )

        return {
            'query_input_ids': query_padded_input_ids,
            'query_embedding_indices': torch.tensor(query_embedding_indices),
            'positive_input_ids': positive_padded_input_ids,
            'positive_embedding_indices': torch.tensor(positive_embedding_indices),
            'negative_input_ids': negative_padded_input_ids,
            'negative_embedding_indices': torch.tensor(negative_embedding_indices),
            'continuation_labels': padded_continuation_labels,
            'whole_input_ids': whole_input_ids,
            'continuation_length': torch.tensor(continuation_length)
        }

    encode_ds = dataset.map(
        tokenize,
        batched=True,
        batch_size=training_args.train_batch_size,
        drop_last_batch=True,
        num_proc=1
    )
    encode_ds.set_format(
        type='torch',
        columns=[
            'query_input_ids', 'query_embedding_indices',
            'positive_input_ids', 'positive_embedding_indices',
            'negative_input_ids', 'negative_embedding_indices',
            'continuation_labels', 'whole_input_ids', 'continuation_length'
        ]
    )
    return encode_ds


class PreprocessedDataset(TorchDataset):
    """Dataset class for loading preprocessed tokenized data."""
    def __init__(self, seed):
        self.data = torch.load('./tokenized_train_data.pt')

    def __len__(self):
        return len(self.data['query_input_ids'])

    def __getitem__(self, idx):
        return {
            'query_input_ids': self.data['query_input_ids'][idx],
            'query_embedding_indices': self.data['query_embedding_indices'][idx],
            'positive_input_ids': self.data['positive_input_ids'][idx],
            'positive_embedding_indices': self.data['positive_embedding_indices'][idx],
            'negative_input_ids': self.data['negative_input_ids'][idx],
            'negative_embedding_indices': self.data['negative_embedding_indices'][idx]
        }


def build_dataset_load_from_disk():
    return PreprocessedDataset(2025)


def all_gather_tensor(tensor):
    """All-gather tensors across distributed processes."""
    world_size = torch.distributed.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


# Set up logger
logger = logging.getLogger(name='my_logger')
os.makedirs('./wiki_train', exist_ok=True)
logging.basicConfig(filename=os.path.join('./wiki_train', f'wiki_{data_args.task_type}_supervised.log'),
                    level=logging.INFO,
                    format='%(name)s - %(levelname)s - %(message)s')

# Load dataset
encode_ds = load_from_disk('tokenized_data')
encode_ds.set_format(type='torch',
                     columns=['query_input_ids', 'query_embedding_indices',
                              'positive_input_ids', 'positive_embedding_indices',
                              'negative_input_ids', 'negative_embedding_indices'])

torch.cuda.set_device(training_args.local_rank)

# Initialize model
model = Llm2Comp_Third_Stage_Bidirection_Gather(data_args, training_args.local_rank)
model.model_wrapper()
checkpoint = torch.load(model_args.checkpoint_path, map_location=lambda storage, loc: storage.cuda())
model.load_state_dict(checkpoint['module'], strict=True)

# Apply LoRA
model.model = model.model.encoder
model.model.merge_and_unload()
lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
    task_type=TaskType.FEATURE_EXTRACTION,
    lora_alpha=32,
    lora_dropout=0.05
)
model.model = get_peft_model(model.model, lora_config)
model.set_attn_dropout(training_args.dropout_rate)
model.model_wrapper_new()

# Show trainable params
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter: {name}\tShape: {param.size()}")

# Initialize DeepSpeed
model_engine, _, _, _ = deepspeed.initialize(
    args=training_args,
    config_params=training_args.deepspeed,
    model=model,
    model_parameters=model.parameters()
)

# Training loop
dataLoader = get_distributed_dataloader(encode_ds, int(training_args.train_batch_size / 8))
model_engine.train()
temperature = 0.05
for epoch in range(training_args.train_epoch):
    for idx, batch in enumerate(tqdm(dataLoader, desc=f'Epoch: {epoch+1}', total=1000)):
        batch = {k: v.cuda() for k, v in batch.items()}
        query_embedding, positive_embedding, negative_embedding = model_engine(**batch)

        # Gather embeddings across GPUs
        full_query_embedding = torch.cat(mismatched_sizes_all_gather(query_embedding))
        full_positive_embedding = torch.cat(mismatched_sizes_all_gather(positive_embedding))
        full_negative_embedding = torch.cat(mismatched_sizes_all_gather(negative_embedding))

        # Contrastive loss
        full_weight_embedding = torch.cat([full_positive_embedding, full_negative_embedding], dim=0)
        dot_products = full_query_embedding @ full_weight_embedding.T
        probs = F.log_softmax(dot_products / temperature, dim=1)
        ground_truth = torch.arange(probs.shape[0]).long().cuda()
        loss = F.nll_loss(probs, ground_truth)

        model_engine.backward(loss)
        current_lr = model_engine.get_lr()[0]
        if training_args.local_rank == 0:
            logger.info(f"Epoch: {epoch+1}, Batch:{idx+1}, Loss: {loss}, LR: {current_lr}")
            wandb.log({"loss": loss})

        if (idx + 1) % 200 == 0:
            model_engine.save_checkpoint(f'{model_args.save_dir}_step_{idx}')
        model_engine.step()

    model_engine.save_checkpoint(f'{model_args.save_dir}_step_{idx}')
