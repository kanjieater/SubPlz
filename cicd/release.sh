#!/bin/bash

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❗ GitHub CLI (gh) is not installed. Please install it from https://cli.github.com/manual/installation"
    exit 1
fi

# Ask for the tag name
read -p "Enter the tag name for the release: " TAG_NAME

# Check if a tag name was provided
if [[ -z "$TAG_NAME" ]]; then
    echo "❗ Tag name cannot be empty. Please provide a valid tag name."
    exit 1
fi

# Get the latest commit hash and message
LATEST_COMMIT=$(git rev-parse HEAD)
LATEST_MESSAGE=$(git log -1 --pretty=%B)

# Create a tag in the local repository
git tag -a "$TAG_NAME" -m "$LATEST_MESSAGE"

# Push the tag to the remote repository
git push origin "$TAG_NAME"

# Create a GitHub release with the tag
gh release create "$TAG_NAME" --title "$TAG_NAME" --notes "$LATEST_MESSAGE"

# Check if the release was successful
if [ $? -eq 0 ]; then
    echo "✅ Release $TAG_NAME created successfully!"
else
    echo "❗ Failed to create the release. Please check your GitHub CLI setup."
    exit 1
fi