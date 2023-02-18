#!/bin/bash
set -e

FOLDER="${1:-pwd}"
# for f in $FOLDER/*.mp3; do echo "file '$f'"; done
# echo <( for f in "$FOLDER/"*.mp3; do echo "file '$f'"; done )
ffmpeg -f concat -safe 0 -i <( for f in "$FOLDER/"*; do echo "file '$f'"; done ) "$FOLDER/audio.mp3"
# ( for f in *.mp3; do echo "file '$(pwd)/$f'"; done )