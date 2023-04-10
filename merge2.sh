#!/bin/bash

audiobooks_path="/mnt/a/audiobooksm4b/"

# Loop over each audiobook folder
for audiobook_folder in "${audiobooks_path}"/*/
do
  # Print out the name of the audiobook folder
  echo "Merging files in ${audiobook_folder}..."

  # Get the audiobook name
  audiobook_name="$(basename "${audiobook_folder}")"

  # Run the m4b-tool Docker container to merge the MP4 files
  docker_cmd="docker run -it --rm -u $(id -u):$(id -g) -v \"${audiobook_folder}\":/mnt sandreas/m4b-tool:latest merge \"./\" --output-file \"./${audiobook_name}.m4b\" --jobs $(nproc --all)"

  echo "$docker_cmd"
  eval "$docker_cmd"
done

