#!/bin/bash

# Get the list of directories from the Python command
directories=$(subplz find -d '/mnt/v/Videos/J-Anime Shows/hells paradise/')

# Check if directories variable is empty
if [[ -z "$directories" ]]; then
  echo "No directories found."
  exit 1
fi

# Iterate over each directory and run the subsequent commands
echo "$directories" | tr -d "[]'" | tr ',' '\n' | while IFS= read -r directory; do
  # Trim any leading or trailing whitespace
  directory=$(echo "$directory" | xargs)

  # Check if the directory contains "Error accessing"
  if [[ "$directory" == *"Error accessing"* ]]; then
    echo "Skipping error message: $directory"
    continue
  fi

  if [[ -n "$directory" ]]; then
    echo "Processing directory: $directory"

    # echo "Renaming files in : $directory"
    # subplz rename -d "$directory" --lang-ext "ab" --lang-ext-original "ja"

    echo "Alass Syncing in : $directory"
    subplz sync -d "$directory" --lang-ext "as" --lang-ext-original "en" --lang-ext-incorrect "ab" --alass

    # echo "SubPlz Syncing in : $directory"
    # subplz sync -d "$directory" --lang-ext "ay" --lang-ext-original "ab"

    # echo "Generating in : $directory"
    # subplz gen -d "$directory" --lang-ext "az" --model large-v3

    # echo "Copying prioritized in : $directory"
    # subplz copy -d "$directory" --lang-ext "ja" --lang-ext-priority "as" "ay" "az" "ab" --overwrite
  else
    echo "Directory name is empty."
  fi
done