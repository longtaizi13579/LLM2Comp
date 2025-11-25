from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class DataTrainingArguments:
    path: Optional[str] = field(
        default='./wiki1m_for_simcse.txt',
    )
    special_token_id: Optional[int] = field(
        default=32000,#128205,
    )
    sequence_length: Optional[int] = field(
        default=8,
    )
    max_length: int = field(default=512)
    min_continuation_length: int = field(default=1)
@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={
            "help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
        },
    )
    save_dir: Optional[str] = field(
        default="KL-first-bidirectional",
        metadata={
            "help": "The place to save the trained model."
        },
    )
@dataclass
class TrainingArguments:
    weight_decay: float = field(default=1e-5)
    lr: float = field(default=1e-4)
    deepspeed:Optional[str] = field(
        default="./deepspeed_stage1.json",
        metadata={
            "help": "Deepspeed file path."
        }
    )
    local_rank: Optional[int] = field(default=0)
    train_epoch: Optional[int] = field(default=1)
    train_batch_size: Optional[int] = field(default=4)
    task_type: Optional[str] = field(default='KL')