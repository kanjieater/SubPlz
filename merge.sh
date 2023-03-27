#!/bin/bash
set -e

FOLDER="${1}"
echo $FOLDER
NAME="$(basename "$FOLDER")"
echo $NAME
SPLIT_DIR="$FOLDER/$(echo $NAME)_merge/"
REL_SPLIT_DIR="./$(echo $NAME)_merge/"
ls -1v "$SPLIT_DIR"


# function m4b-tool() {
#   command 'docker run -it --rm -u $(id -u):$(id -g) -v "$(pwd)":/mnt sandreas/m4b-tool:latest'
# }
# cd $FOLDER
# echo $(pwd)
# m4b-tool merge "$FOLDER" --output-file="$FOLDER/merged.m4b"
# ls -1 "$SPLIT_DIR"*.{mp4,mp3,m4b} > "$SPLIT_DIR/files.txt"
# $FILES="$(cat $SPLIT_DIR/files.txt)"
# echo $FILES
docker run -it --rm -u $(id -u):$(id -g) -v "$FOLDER":/mnt sandreas/m4b-tool:latest merge "$REL_SPLIT_DIR"\
 --output-file="./$NAME.m4b" --jobs $(nproc --all) --album "$NAME" --name "$NAME" --use-filenames-as-chapters
