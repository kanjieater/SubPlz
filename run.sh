#!/bin/bash
set -e


FOLDER="${1:-pwd}"
SCRIPTNAME="${2:-script.txt}"
TIMINGSUBS="${3:-timings.vtt}"
echo "Script: $FOLDER/$SCRIPTNAME";

NAME="$(basename "$FOLDER" )"

# Cleaning any previous runs data
rm -f "$FOLDER/matched.vtt"
rm -f "$FOLDER/timings.vtt"
rm -f "$FOLDER/$SCRIPTNAME.split.txt"

python split_run.py "$FOLDER/"
python split-sentences.py "$FOLDER/$SCRIPTNAME"
python align_vtt_v2.py "$FOLDER/$SCRIPTNAME.split.txt" "$FOLDER/$TIMINGSUBS" "$FOLDER/matched.vtt" --mode 2
ffmpeg -y -c:s subrip -i "$FOLDER/matched.vtt" "$FOLDER/$NAME.srt" -hide_banner -loglevel error
rm "$FOLDER/matched.vtt"
rm "$FOLDER/timings.vtt"
rm -f "$FOLDER/$NAME.vtt"
rm -f "$FOLDER/$NAME.ass"
rm -f "$FOLDER/$NAME.filtered.m4b"
rm "$FOLDER/$SCRIPTNAME.split.txt"