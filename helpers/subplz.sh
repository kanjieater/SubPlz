#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <path> [--initial-rename]"
  exit 1
fi

# Get the directory path from the first argument
directories="$1"

# Check for optional second argument
initial_rename=false
if [ "$2" == "--initial-rename" ]; then
  initial_rename=true
fi

# Check if directories variable is empty
if [[ -z "$directories" ]]; then
  echo "No directories found."
  exit 1
fi

# Iterate over each directory and run the subsequent commands
echo "$directories" | tr -d "[]'" | tr ',' '\n' | while IFS= read -r directory; do
  # Trim any leading or trailing whitespace
  directory=$(echo "$directory" | xargs)

  # Add leading slash back if it's missing
  if [[ "$directory" != /* ]]; then
    directory="/$directory"
  fi

  # Check if the directory contains "Error accessing"
  if [[ "$directory" == *"Error accessing"* ]]; then
    echo "Skipping error message: $directory"
    continue
  fi

  if [[ "$initial_rename" == true ]]; then
    echo "Performing initial rename (prep) in: $directory"
    subplz rename -d "$directory" --lang-ext "ab"
  fi

  if [[ -n "$directory" ]]; then
    echo "Processing directory: $directory"
    echo "Renaming files in : $directory"
    subplz rename -d "$directory" --lang-ext "ab" --lang-ext-original "old"
    subplz rename -d "$directory" --lang-ext "ab" --lang-ext-original "ja"

    echo "Alass Syncing in : $directory"
    subplz sync -d "$directory" --lang-ext "as" --lang-ext-original "en" --lang-ext-incorrect "ab" --alass

    echo "SubPlz Syncing in : $directory"
    subplz sync -d "$directory" --lang-ext "ak" --lang-ext-original "ab" --model large-v3

    echo "Generating in : $directory"
    subplz gen -d "$directory" --lang-ext "az" --model large-v3

    echo "Copying prioritized in : $directory"
    subplz copy -d "$directory" --lang-ext "ja" --lang-ext-priority "as" "ak" "az" "ab" --overwrite
  else
    echo "Directory name is empty."
  fi
done
