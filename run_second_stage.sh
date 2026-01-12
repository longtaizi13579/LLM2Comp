export HF_ENDPOINT="https://hf-mirror.com"
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
deepspeed --num_gpus 1 unsupervised_contrastive_learning.py