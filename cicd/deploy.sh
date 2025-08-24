#!/bin/bash

# Variables
REMOTE_IMAGE="kanjieater/subplz:latest"

# Log in to Docker Hub
echo "Logging in to Docker Hub..."
docker login || { echo "Docker login failed"; exit 1; }

# Push the newly tagged image to Docker Hub
echo "Pushing the image to Docker Hub..."
docker push $REMOTE_IMAGE || { echo "Failed to push the image"; exit 1; }

