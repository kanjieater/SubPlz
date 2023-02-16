#!/bin/bash


FOLDER="${1:-pwd}"
SCRIPTNAME="${2:-script.txt}"
TIMINGSUBS="${3:-captions.vtt}"
echo "Script: $FOLDER/$SCRIPTNAME";
echo "Timing Subs: $FOLDER/$TIMINGSUBS";

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate # relies on .python-version

# python split-sentences.py "$FOLDER/$SCRIPTNAME"
# ffmpeg -i "$FOLDER/captions.srt" "$FOLDER/captions.vtt"
# python align-vtt-v2.py "$FOLDER/$SCRIPTNAME.split.txt" "$FOLDER/$TIMINGSUBS" "$FOLDER/done.vtt"
ffmpeg -c:s subrip -i "$FOLDER/done.vtt" "$FOLDER/done.srt"