FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
ARG USERNAME
ARG USER_UID
ARG USER_GID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

USER $USERNAME