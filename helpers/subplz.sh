#!/bin/bash

# Get the list of directories from the Python command
directories=$(subplz find -d '/mnt/v/Videos/J-Anime Shows/' '/mnt/v/Videos/J-Anime Movies/')

# Iterate over each directory and run the subsequent commands
echo "$directories" | while IFS= read -r directory; do
  echo "Processing directory: $directory"
  # subplz rename -d "$directory" --lang-ext "ab" --lang-ext-original "ja"
  # subplz sync -d "$directory" --lang-ext "as" --lang-ext-original "ja" --alass
  # subplz sync -d "$directory" --lang-ext "ay" --lang-ext-original "ja"
  # subplz gen -d "$directory" --lang-ext "az" --model large-v3
  # subplz copy -d "$directory" --lang-ext "ja" --lang-ext-originals "ab" "as" "ay" "az"
done
