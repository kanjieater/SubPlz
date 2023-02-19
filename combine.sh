#!/bin/bash
set -e

FOLDER="${1:-pwd}"
echo $FOLDER
NAME="$(basename "$FOLDER")"
# echo $NAME
# shopt -s expand_aliases
# source ~/.aliases
ls "$FOLDER/$(echo $NAME)_splitted/"

# for f in $FOLDER/*.mp3; do echo "file '$f'"; done
# echo <( for f in "$FOLDER/"*.mp3; do echo "file '$f'"; done )


# function m4b-tool() {
#   command 'docker run -it --rm -u $(id -u):$(id -g) -v "$(pwd)":/mnt sandreas/m4b-tool:latest'
# }
# cd $FOLDER
# echo $(pwd)
# m4b-tool merge "$FOLDER" --output-file="$FOLDER/merged.m4b"

docker run -it --rm -u $(id -u):$(id -g) -v "$FOLDER":/mnt sandreas/m4b-tool:latest merge "./$(echo $NAME)_splitted/" --output-file="./$NAME.m4b" --jobs $(nproc --all)

# ffmpeg -f concat -safe 0 -i <( for f in "$FOLDER/"*.mp3; do echo "file '$f'"; done ) -map_metadata 0 -c copy -write_id3v1 true -id3v2_version 0 "$FOLDER/audio.mp3"
# ffmpeg -f concat -safe 0 -i <( for f in "$FOLDER/"*.mp3; do echo "file '$f'"; done ) -c copy -filter_complex "color[c];[c][0]scale2ref[c][art];[c][art]overlay" -shortest "$FOLDER/audio.mp3"
# mp3wrap "$FOLDER/audio.mp3" "$FOLDER/*.mp3"
# ( for f in *.mp3; do echo "file '$(pwd)/$f'"; done )