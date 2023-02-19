#!/bin/bash
set -e


FOLDER="${1:-pwd}"
SCRIPTNAME="${2:-script.txt}"
TIMINGSUBS="${3:-timings.vtt}"
echo "Script: $FOLDER/$SCRIPTNAME";
echo "Timing Subs: $FOLDER/$TIMINGSUBS";

NAME="$(basename "$FOLDER" )"
# SPLIT_NAME="$FOLDER/$(echo "$NAME")_splitted/"
# echo $SPLIT_NAME

python split_run.py "$FOLDER"

python split-sentences.py "$FOLDER/$SCRIPTNAME"
python align_vtt_v2.py "$FOLDER/$SCRIPTNAME.split.txt" "$FOLDER/$TIMINGSUBS" "$FOLDER/matched.vtt" --mode 2
ffmpeg -y -c:s subrip -i "$FOLDER/matched.vtt" "$FOLDER/$NAME.srt"
rm "$FOLDER/matched.vtt"
rm "$FOLDER/timings.vtt"
rm "$FOLDER/$SCRIPTNAME.split.txt"