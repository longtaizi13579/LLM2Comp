from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class DataTrainingArguments:
    path: Optional[str] = field(
        default='e5-dataset'
    )
    special_token_id: Optional[int] = field(
        default=32000, #128205
    )
    sequence_length: Optional[int] = field(
        default=8,
    )
    max_length: int = field(default=512)
    min_continuation_length: int = field(default=1)
    
@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", #"mistralai/Mistral-7B-v0.1",
        metadata={
            "help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
        },
    )
    checkpoint_path: Optional[str] = field(
        default="./mp_rank_00_model_states.pt",
        metadata={
            "help": "The trained model checkpoint to load."
        },
    )
    save_dir: Optional[str] = field(
        default="NLL-third-bidirectional",
        metadata={
            "help": "The place to save the trained model."
        },
    )

@dataclass
class TrainingArguments:
    weight_decay: float = field(default=1e-4)
    lr: float = field(default=1e-4)
    deepspeed:Optional[str] = field(
        default="./deepspeed_stage3.json",
        metadata={
            "help": "Deepspeed file path."
        }
    )
    local_rank: Optional[int] = field(default=0)
    train_epoch: Optional[int] = field(default=1)
    train_batch_size: Optional[int] = field(default=1024)
    dropout_rate: float = field(default=0.0)
    task_type: Optional[str] = field(default='KL')