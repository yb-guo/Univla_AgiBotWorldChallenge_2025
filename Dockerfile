# 基础镜像：CUDA 12.1 + cuDNN8 + Ubuntu 22.04
FROM singularitybase.azurecr.io/base/job/pytorch/acpt-2.2.2-py3.10-cuda12.1:20250226T224107506

# 设置时区，防止交互阻塞
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 更新系统并安装常用依赖
RUN apt-get update && apt-get install -y \
    git wget curl unzip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
      fuse3 curl ca-certificates tar \
  && rm -rf /var/lib/apt/lists/*
  
# 安装 Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# 配置 Microsoft 包仓库并安装 blobfuse2
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release; \
    # 下载并安装 Microsoft packages signing key + repo
    curl -fsSL https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -o /tmp/packages-microsoft-prod.deb; \
    dpkg -i /tmp/packages-microsoft-prod.deb; \
    rm -f /tmp/packages-microsoft-prod.deb; \
    apt-get update; \
    # 然后直接安装 blobfuse2（包名：blobfuse2）
    apt-get install -y --no-install-recommends blobfuse2 fuse3; \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /work

# 容器启动默认进入 bash
CMD ["/bin/bash"]
