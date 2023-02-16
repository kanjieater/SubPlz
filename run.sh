#!/bin/bash


FOLDER="${1:-pwd}"
SCRIPTNAME="${2:-script.txt}"
TIMINGSUBS="${3:-captions.vtt}"
echo "Script: $FOLDER/$SCRIPTNAME";
echo "Timing Subs: $FOLDER/$TIMINGSUBS";

# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"
# pyenv activate # relies on .python-version

#prep
# ffmpeg -i "$FOLDER/1.mp3" -ar 16000 -ac 1 "$FOLDER/1.wav"

stable-ts "$FOLDER/1.wav" --language Japanese --output_dir "$FOLDER/" --model large-v2 -o "$FOLDER/captions.ass"

python split-sentences.py "$FOLDER/$SCRIPTNAME"
ffmpeg -y -i "$FOLDER/captions.ass" "$FOLDER/captions.vtt"
python align-vtt-v2.py "$FOLDER/$SCRIPTNAME.split.txt" "$FOLDER/$TIMINGSUBS" "$FOLDER/done.vtt" --mode 1
ffmpeg -y -c:s subrip -i "$FOLDER/done.vtt" "$FOLDER/done.srt"