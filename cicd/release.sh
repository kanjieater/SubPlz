#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. VALIDATE INPUT ---
# Check if a version tag was provided as an argument.
if [[ -z "$1" ]]; then
    echo "❌ ERROR: No version tag provided."
    echo "Usage: ./cicd/create-release.sh <version_tag>"
    echo "Example: ./cicd/create-release.sh v4.0.0"
    exit 1
fi

# --- 2. DEFINE VARIABLES ---
VERSION_TAG="$1"
LOCAL_IMAGE_NAME="subplz"
REMOTE_REPO="kanjieater/subplz"
DOCKERFILE_PATH="."

echo "🚀 Starting release process for version: $VERSION_TAG"

# --- 3. DOCKER BUILD & TAG ---
echo "📦 Building the Docker image..."
docker build -t "$LOCAL_IMAGE_NAME:latest" "$DOCKERFILE_PATH"

echo "🏷️ Tagging Docker image with '$REMOTE_REPO:latest' and '$REMOTE_REPO:$VERSION_TAG'..."
docker tag "$LOCAL_IMAGE_NAME:latest" "$REMOTE_REPO:latest"
docker tag "$LOCAL_IMAGE_NAME:latest" "$REMOTE_REPO:$VERSION_TAG"

# --- 4. DOCKER PUSH ---
echo "Logging in to Docker Hub..."
docker login

echo "🚢 Pushing both 'latest' and '$VERSION_TAG' tags to Docker Hub..."
docker push "$REMOTE_REPO:latest"
docker push "$REMOTE_REPO:$VERSION_TAG"
echo "✅ Docker push complete."

# --- 5. GIT & GITHUB RELEASE ---
echo "🔖 Creating Git tag and GitHub release..."

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❗ GitHub CLI (gh) is not installed. Please install it to create a release."
    exit 1
fi

# Get the latest commit message for the release notes
LATEST_MESSAGE=$(git log -1 --pretty=%B)

echo "Creating git tag '$VERSION_TAG'..."
git tag -a "$VERSION_TAG" -m "$LATEST_MESSAGE"

echo "Pushing git tag to remote..."
git push origin "$VERSION_TAG"

echo "Creating GitHub release..."
gh release create "$VERSION_TAG" --title "$VERSION_TAG" --notes "$LATEST_MESSAGE"

echo "🎉 Release $VERSION_TAG created successfully!"