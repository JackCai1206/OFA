#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1097

log_dir=./sgcls_logs
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
mkdir -p $log_dir

data_dir=../../dataset/OFA_data/sgcls
data=${data_dir}/vg_val_full.tsv
path=../../checkpoints/OFA/sgcls_checkpoints/_20_3e-5_512/tmp/checkpoint8.pt
result_path=../../results/sgcls
split='valid'

log_dir=./sgcls_logs
log_file=${log_dir}/eval_epoch_20_beam_5_min-len_80.log

# Check to not override log file
if [ -f $log_file ]; then
    echo "Log file ${log_file} already exists, aborting..."
    exit 1
fi

CUDA_VISIBLE_DEVICES=3 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=sgcls \
    --batch-size=48 \
    --log-format=simple --log-interval=5 \
    --seed=7 \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=1000 \
    --min-len=80 \
    --no-repeat-ngram-size=6 \
    --fp16 \
    --num-workers=0 \
    --vg-json-dir=../../dataset/visual_genome/VG-SGG-dicts-with-attri.json \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False}" > ${log_file} 2>&1
