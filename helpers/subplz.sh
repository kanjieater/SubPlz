#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <path_to_process> [--initial-rename]"
  exit 1
fi

# Get the single directory path from the first argument
# The shell has already handled unquoting and escapes.
# directory_to_process now holds the correct, full path.
directory_to_process="$1"

# Check for optional second argument
initial_rename=false
if [ "$2" == "--initial-rename" ]; then
  initial_rename=true
fi

# Check if directory_to_process variable is empty (should be caught by $# -lt 1)
if [[ -z "$directory_to_process" ]]; then
  echo "No directory path provided."
  exit 1
fi

# No need for the loop or tr commands if we're processing a single path.

# Check if the directory contains "Error accessing" (this check might be for specific input formats)
if [[ "$directory_to_process" == *"Error accessing"* ]]; then
  echo "Skipping what appears to be an error message: $directory_to_process"
  exit 0 # Exit cleanly if this is an error condition to skip
fi

# --- Now, use "$directory_to_process" in all your subplz commands ---

if [[ "$initial_rename" == true ]]; then
  echo "Performing initial rename (prep) in: $directory_to_process"
  subplz rename -d "$directory_to_process" --lang-ext "ab"
fi

# No need for 'if [[ -n "$directory" ]]' if we've already validated it.
echo "Processing directory: $directory_to_process"

echo "Renaming files in : $directory_to_process"
subplz rename -d "$directory_to_process" --lang-ext "ab" --lang-ext-original "old"
subplz rename -d "$directory_to_process" --lang-ext "ab" --lang-ext-original "ja"

echo "Extracting & Verifying Native Target Language subs if they exist"
subplz extract -d "$directory_to_process" --lang-ext "na" --lang-ext-original "ja"

echo "Alass Syncing in : $directory_to_process"
subplz sync -d "$directory_to_process" --lang-ext "as" --lang-ext-original "en" --lang-ext-incorrect "ab" --alass

echo "SubPlz Syncing in : $directory_to_process"
subplz sync -d "$directory_to_process" --lang-ext "ak" --lang-ext-original "ab" --model large-v3

echo "Generating in : $directory_to_process"
subplz gen -d "$directory_to_process" --lang-ext "az" --model large-v3

echo "Copying prioritized in : $directory_to_process"
subplz copy -d "$directory_to_process" --lang-ext "ja" --lang-ext-priority "na" "as" "ak" "az" "ab" --overwrite

echo "All operations completed for: $directory_to_process"