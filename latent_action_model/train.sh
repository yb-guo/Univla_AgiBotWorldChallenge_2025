export WANDB_MODE=offline
# export NCCL_SOCKET_IFNAME=eth0

cd /mnt/chenjin/AgiBot-World
source /home/chenjin/miniconda3/etc/profile.d/conda.sh
conda activate vla

torchrun \
--standalone \
--nnodes 1 \
--nproc-per-node 8 \
/mnt/chenjin/AgiBot-World/latent_action_model/main.py fit \
--config /mnt/chenjin/AgiBot-World/latent_action_model/config/lam-stage-2.yaml \
2>&1 | tee /mnt/chenjin/AgiBot-World/latent_action_model/lam-a2d-stage-2.log
