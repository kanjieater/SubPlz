# --- Stage 1: The Dependency Builder ---
FROM python:3.11-slim as builder

WORKDIR /app

# Install all dependencies based on pyproject.toml
COPY pyproject.toml ./
RUN pip install --no-cache-dir .[automation]

# --- Stage 2: The Final Production Image ---
FROM lscr.io/linuxserver/faster-whisper:gpu-v2.4.0-ls72

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set the working directory and change ownership
WORKDIR /app
RUN chown -R appuser:appuser /app

# Switch to non-root user for dependency installation
USER appuser

# Add user's local bin to PATH for pip-installed executables
ENV PATH="/home/appuser/.local/bin:$PATH"

# --- Fixed Dependency Installation ---
# Copy the requirements and install directly in the PyTorch image
# Copy the requirements and install directly in the PyTorch image
COPY pyproject.toml ./
RUN pip install --no-cache-dir .[automation]

# Copy your application's source code
COPY . .

# Install the subplz package itself in editable mode
RUN pip install --no-cache-dir -e . --no-deps
# --- End Installation ---

# Set the entrypoint to your application
ENTRYPOINT ["subplz"]

# Set a default command to run
CMD ["--help"]