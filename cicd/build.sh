#!/bin/bash

# Variables
LOCAL_IMAGE_NAME="subplz"
REMOTE_IMAGE="kanjieater/subplz:latest"
DOCKERFILE_PATH="."

# Build the Docker image
echo "Building the Docker image..."
docker build -t $LOCAL_IMAGE_NAME:latest $DOCKERFILE_PATH || { echo "Docker build failed"; exit 1; }

# Tag the local image with the new name
echo "Tagging the local image..."
docker tag $LOCAL_IMAGE_NAME:latest $REMOTE_IMAGE || { echo "Docker tag failed"; exit 1; }

echo "Build and tag complete."
