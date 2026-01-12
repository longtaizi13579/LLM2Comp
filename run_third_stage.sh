export HF_ENDPOINT="https://hf-mirror.com"
deepspeed --num_gpus 8 supervised_contrastive_learning.py --train_batch_size 1024 --path ./data