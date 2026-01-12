export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO 
deepspeed --num_gpus 1 pretext_compression.py --sequence_length 8 --task_type KL