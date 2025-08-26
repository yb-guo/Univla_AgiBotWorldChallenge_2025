FROM singularitybase.azurecr.io/base/job/pytorch/acpt-torch2.5.0-py3.10-cuda12.4-ubuntu22.04:20250227T132634623
# RUN apt-get update
# RUN apt install -y wget
# RUN apt install -y git
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_25.1.1-2-Linux-x86_64.sh
# RUN chmod +x Miniconda3-py310_25.1.1-2-Linux-x86_64.sh
# RUN ./Miniconda3-py310_25.1.1-2-Linux-x86_64.sh -b
# RUN rm Miniconda3-py310_25.1.1-2-Linux-x86_64.sh
# ENV PATH="/root/miniconda3/bin:${PATH}"
# ENV WORKDIR="/work"
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
# RUN dpkg -i cuda-keyring_1.1-1_all.deb
# RUN apt-get update
# RUN apt-get -y install cudnn-cuda-12
# RUN rm cuda-keyring_1.1-1_all.deb
# RUN source ~/miniconda3/bin/activate
# RUN pip install packaging ninja
# RUN pip install accelerate>=0.25.0 draccus>=0.8.0 einops huggingface_hub json-numpy jsonlines matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 \
#     timm==0.9.10 tokenizers==0.13.3 transformers==4.31.0  tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3

# RUN pip install "flash-attn==2.5.5" --no-build-isolation
# RUN pip install opencv-python
# RUN apt-get update
# RUN apt-get -y install libglib2.0-0
RUN pip install wandb
ENV WANDB_API_KEY="0a85800ab1d195b8ca06aeda0a8c04efc5d917d5"
RUN wandb login --host=https://microsoft-research.wandb.io
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

RUN wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb
RUN dpkg -i packages-microsoft-prod.deb
RUN apt-get update
RUN apt-get install libfuse3-dev fuse3 -y
RUN apt-get install blobfuse2 -y
RUN apt-get install -y cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev




WORKDIR /work