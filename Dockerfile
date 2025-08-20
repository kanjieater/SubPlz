# Use the LinuxServer.io base image
FROM ghcr.io/linuxserver/faster-whisper:gpu-2.0.0-ls23

# Set environment variables
ENV PUID=1000 \
    PGID=1000 \
    TZ=Etc/UTC \
    WHISPER_MODEL=tiny-int8 \
    WHISPER_BEAM=1 \
    WHISPER_LANG=ja

# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       ffmpeg \
       libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY . .
RUN pip install --no-cache-dir --no-deps .

# Create dedicated user (uid 1000), fix ownership
# RUN useradd -m -u 1000 -s /bin/bash subplz \
#     && mkdir -p /config /home/subplz/.cache \
#     && chown -R subplz:subplz /config /app /home/subplz/.cache
RUN useradd -m -u 1000 -s /bin/bash sp
# Switch to non-root user
USER sp

# Ensure the path to subplz is available
ENV PATH="/home/sp/.local/bin:$PATH" \
    BASE_PATH=/config \
    TORCH_HOME=/config/cache/torch \
    XDG_CACHE_HOME=/config/cache/xdg/.cache

# https://github.com/linuxserver/docker-faster-whisper/issues/15
# https://github.com/SYSTRAN/faster-whisper/issues/516
# This probably gets fixed on python 3.12 and a modern faster-whisper image
ENV LD_LIBRARY_PATH="/lsiopy/lib/python3.10/site-packages/nvidia/cublas/lib:/lsiopy/lib/python3.10/site-packages/nvidia/cudnn/lib"

ENTRYPOINT ["subplz"]
CMD ["watch", "--config", "/config/config.yml"]