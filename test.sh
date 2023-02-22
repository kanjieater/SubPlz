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

# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"
# pyenv activate # relies on .python-version

# ffmpeg -i "$FOLDER/1.mp3" -ar 16000 -ac 1 "$FOLDER/1.wav"
# whisperx "/mnt/d/Editing/Audiobooks/かがみの孤城/i/1.wav" --language ja --output_dir "/mnt/d/Editing/Audiobooks/かがみの孤城/" --model large-v2 --vad_filter --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --hf_token $HF
# whisper "$FOLDER/audio.mp3" --language ja --model large --output_dir "$FOLDER"

# Main Procedure
stable-ts "$FOLDER/$NAME.mp3" --language ja --output_dir "$FOLDER/" --model large-v2 -o "$FOLDER/timings.ass" --overwrite

# ffmpeg -y -i "$FOLDER/timings.ass" "$FOLDER/timings.vtt"

# python split-sentences.py "$FOLDER/$SCRIPTNAME"
# python align_vtt_v2.py "$FOLDER/$SCRIPTNAME.split.txt" "$FOLDER/$TIMINGSUBS" "$FOLDER/done.vtt" --mode 2
# ffmpeg -y -c:s subrip -i "$FOLDER/done.vtt" "$FOLDER/audio.srt"


python split_run.py "$FOLDER"

python split-sentences.py "$FOLDER/$SCRIPTNAME"
python align_vtt_v2.py "$FOLDER/$SCRIPTNAME.split.txt" "$FOLDER/$TIMINGSUBS" "$FOLDER/matched.vtt" --mode 2
ffmpeg -y -c:s subrip -i "$FOLDER/matched.vtt" "$FOLDER/$NAME.srt"
rm "$FOLDER/matched.vtt"
rm "$FOLDER/timings.vtt"
rm "$FOLDER/$SCRIPTNAME.split.txt"