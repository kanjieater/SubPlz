# Use the LinuxServer.io base image
FROM ghcr.io/linuxserver/faster-whisper:2.0.0-gpu

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

COPY pyproject.toml /tmp/
RUN pip install --no-cache-dir /tmp/

# https://github.com/linuxserver/docker-faster-whisper/issues/15
# https://github.com/SYSTRAN/faster-whisper/issues/516
ENV LD_LIBRARY_PATH="/lsiopy/lib/python3.10/site-packages/nvidia/cublas/lib:/lsiopy/lib/python3.10/site-packages/nvidia/cudnn/lib"

COPY . /app

# Set work directory
WORKDIR /app

# # Expose port for Wyoming connection
# EXPOSE 10300

# Healthcheck to ensure container is running
HEALTHCHECK --interval=30s --timeout=10s CMD nc -z localhost 10300 || exit 1

# Start the application
ENTRYPOINT ["python", "-m", "subplz"]
CMD ["--help"]