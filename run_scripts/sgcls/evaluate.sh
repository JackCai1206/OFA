#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1091

log_dir=./sgcls_logs
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
mkdir -p $log_dir

data_dir=../../dataset/OFA_data/sgcls
data=${data_dir}/vg_val_full.tsv
path=../../checkpoints/OFA/sgcls_checkpoints/_12_3e-5_512/checkpoint_last.pt
result_path=../../results/sgcls
split='valid'

log_dir=./sgcls_logs
log_file=${log_dir}/eval_${max_epoch}"_"${lr}"_"${patch_image_size}".log"

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=sgcls \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=200 \
    --no-repeat-ngram-size=6 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False}" > ${log_file} 2>&1
