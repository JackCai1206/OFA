#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1098

log_dir=./vrd_logs
user_dir=../../ofa_module
bpe_dir=../../utils/BPE
mkdir -p $log_dir

data_dir=../../dataset/OFA_data/sgcls
data=${data_dir}/vg_val_full.tsv
path=../../checkpoints/OFA/vrd_checkpoints/_16_3e-5_512_rare_noempty_balanced_tgtbox_3/checkpoint4.pt
# path=../../checkpoints/OFA/vrd_checkpoints/_20_3e-5_512/tmp/checkpoint_last.pt
result_path=../../results/vrd
split='valid'

log_dir=./vrd_logs
log_file=${log_dir}/eval_epoch_4_beam_5_min-len_15_3e-5_noempty_balanced_tgtbox_3.log

# Check to prompt confirm override log file
if [ -f $log_file ]; then
    read -p "Log file ${log_file} already exists, override? (y/n): " confirm
    if [ $confirm != "y" ]; then
        exit 1
    fi
fi

CUDA_VISIBLE_DEVICES=3 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vrd \
    --batch-size=4 \
    --log-format=simple --log-interval=5 \
    --seed=7 \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=100 \
    --min-len=15 \
    --no-repeat-ngram-size=6 \
    --fp16 \
    --num-workers=0 \
    --vg-json-dir=../../dataset/visual_genome/VG-SGG-dicts-with-attri.json \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False}" > ${log_file} 2>&1
