conda create -n univla_merged python=3.10 -y
conda activate univla_merged

# PyTorch 2.2.0 + CUDA 12.1 官方安装命令
pip install torch==2.2.0 torchvision==0.17.0 --force-reinstall

# numpy 
pip install numpy==1.26.4

# 安装系统工具和编译依赖
pip install --upgrade pip setuptools wheel ninja

# 安装 xformers，优先 pip
pip install xformers==0.0.24

# 安装 dinov2 依赖（跳过cuml）
pip install omegaconf torchmetrics==0.10.3 fvcore iopath submitit cuml-cu12

# 安装 internvl 依赖
pip install -r requirements/internvl_chat.txt

# 安装 univla 依赖（直接装requirements.txt）
pip install -r requirements.txt

# 安装 dinov2源码包
cd Depth-Anything/torchhub/facebookresearch_dinov2_main
pip install -e .

requirement.txt
omegaconf
torchmetrics==0.10.3
fvcore
iopath
submitit

# 其他
pip install wandb
pip install einops_exts
pip install imgaug
 