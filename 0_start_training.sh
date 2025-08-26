#!/bin/bash

export WANDB_API_KEY="3cc7bfe0dbc951899ed08bf713e010eeebd45365"
export WANDB_MODE="online"   # 如果想禁用，可以设为 "offline"
export PYTHONPATH='/work'
which python

echo " ######## Oral in train.sh ########"
echo "当前 Python: $(which python)"
echo "当前 pip: $(which pip)"
echo " ########"

# 自动登录 W&B
if command -v wandb &> /dev/null; then
    echo "Logging into Weights & Biases..."
    wandb login --relogin $WANDB_API_KEY
else
    echo "wandb 命令不存在，请先安装: pip install wandb"
    exit 1
fi

# 检查 tensorboard 是否安装
# if ! python -c "import tensorboard" &> /dev/null; then
#     echo "tensorboard 未安装，正在安装..."
#     # 优先用 conda 安装，如果没有 conda 再用 pip
#     pip install tensorboard --user
# else
#     echo "tensorboard 已安装"
# fi

# 提取所有 task 文件里的数字
task_ids=$(ls /work/data/Univla/Manipulation-SimData/meta_data_info/task_*json | grep -oP 'task_\K[0-9]+' | sort -n | uniq)
echo "Task IDs: $task_ids"
task_ids_str=$(echo $task_ids | tr '\n' ' ')
echo "Task IDs string: $task_ids_str"

torchrun \
--standalone \
--nnodes 1 \
--nproc-per-node 1 \
scripts/finetune.py \
--vla_path /work/weights/agibot_3rd_ckpt/univla/univla-7b \
--lam_path /work/weights/agibot_3rd_ckpt/univla/univla-latent-action-model/lam-stage-2.ckpt \
--data_root_dir /work/data/Univla/Manipulation-SimData/_extracted \
--meta_json_dir /work/data/Univla/Manipulation-SimData/meta_data_info \
--codebook_size 16 \
--batch_size 4 \
--grad_accumulation_steps 1 \
--max_steps 10000 \
--save_steps 1000 \
--decoder_n_layers 2 \
--decoder_hidden_dim 1024 \
--run_root_dir outputs/train_univla \
--adapter_tmp_dir outputs/train_univla \
--save_latest_checkpoint_only \
--with_proprio \
--use_lora \
--task_ids $task_ids_str