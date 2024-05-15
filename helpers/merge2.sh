#!/bin/bash

audiobooks_path="/mnt/a/MusicBee/test"

# Loop over each audiobook folder
for audiobook_folder in "${audiobooks_path}"/*/
do
  # Print out the name of the audiobook folder
  echo "Merging files in ${audiobook_folder}..."

  # Get the audiobook name
  audiobook_name="$(basename "${audiobook_folder}")"

  # Get the list of mp4 files in the audiobook folder
  mp4_files=$(ls -1v "${audiobook_folder}"*.mp3)

  # Replace audiobooks_path with "./" in each mp4 file path
  mp4_files=$(echo "$mp4_files" | sed "s|${audiobook_folder}|./|g")

  # Wrap each mp4 file path in quotes
  mp4_files=$(echo "$mp4_files" | awk '{printf "\"%s\" ", $0}')

  # Run the m4b-tool Docker container to merge the MP4 files
  docker_cmd="docker run -it --rm -u $(id -u):$(id -g) -v \"${audiobook_folder}\":/mnt sandreas/m4b-tool:latest merge ${mp4_files} --output-file \"./${audiobook_name}.m4b\" --jobs $(nproc --all) --name \"${audiobook_name}\""

  echo "$docker_cmd"
  eval "$docker_cmd"
done