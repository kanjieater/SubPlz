# Use the LinuxServer.io base image
FROM ghcr.io/linuxserver/faster-whisper:gpu

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

#remove when we get abts updated w/ a pyproject.toml
COPY /home/ke/code/AudiobookTextSync /tmp/AudiobookTextSync

COPY pyproject.toml /tmp/
RUN pip install --no-cache-dir /tmp/

# Copy application files
COPY . /app

# Set work directory
WORKDIR /app

# # Expose port for Wyoming connection
# EXPOSE 10300

# Healthcheck to ensure container is running
HEALTHCHECK --interval=30s --timeout=10s CMD nc -z localhost 10300 || exit 1

# Start the application
# ENTRYPOINT ["python", "./src/main.py"]
# CMD ["--help"]