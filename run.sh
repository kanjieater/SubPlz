#!/bin/bash
set -e


FOLDER="${1:-pwd}"
SCRIPTNAME="${2:-script.txt}"
TIMINGSUBS="${3:-timings.vtt}"
echo "Script: $FOLDER/$SCRIPTNAME";
echo "Timing Subs: $FOLDER/$TIMINGSUBS";

# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"
# pyenv activate # relies on .python-version

#prep
# ffmpeg -i "$FOLDER/1.mp3" -ar 16000 -ac 1 "$FOLDER/1.wav"

stable-ts "$FOLDER/audio.mp3" --language ja --output_dir "$FOLDER/" --model large-v2 -o "$FOLDER/timings.ass" --overwrite

# whisperx "/mnt/d/Editing/Audiobooks/かがみの孤城/i/1.wav" --language ja --output_dir "/mnt/d/Editing/Audiobooks/かがみの孤城/" --model large-v2 --vad_filter --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --hf_token $HF

# whisper "$FOLDER/audio.mp3" --language ja --model large --output_dir "$FOLDER"

python split-sentences.py "$FOLDER/$SCRIPTNAME"
ffmpeg -y -i "$FOLDER/timings.ass" "$FOLDER/timings.vtt"
python align-vtt-v2.py "$FOLDER/$SCRIPTNAME.split.txt" "$FOLDER/$TIMINGSUBS" "$FOLDER/done.vtt" --mode 2
ffmpeg -y -c:s subrip -i "$FOLDER/done.vtt" "$FOLDER/audio.srt"