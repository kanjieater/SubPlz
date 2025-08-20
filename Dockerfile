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


RUN adduser --disabled-password --gecos "" myuser

RUN mkdir -p /sub_config/ && chown -R myuser:myuser /sub_config && chown -R myuser:myuser /app


USER myuser


# Ensure the path to subplz is available
ENV PATH="/home/myuser/.local/bin:$PATH" \
    BASE_PATH=/sub_config

# https://github.com/linuxserver/docker-faster-whisper/issues/15
# https://github.com/SYSTRAN/faster-whisper/issues/516
# This probably gets fixed on python 3.12 and a modern faster-whisper image
ENV LD_LIBRARY_PATH="/lsiopy/lib/python3.10/site-packages/nvidia/cublas/lib:/lsiopy/lib/python3.10/site-packages/nvidia/cudnn/lib"

ENTRYPOINT ["subplz"]
CMD ["watch", "--config", "sub_config/config.yml"]