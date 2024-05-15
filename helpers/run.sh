#!/bin/bash

# Define variables
audio_filename="/mnt/d/sync/変な家/変な家.m4b"
text_filename="/mnt/d/sync/変な家/変な家.epub"
language="ja"
model="tiny"
threads=32
output_dir="/mnt/d/sync/変な家"
output_format="srt"

# Execute the command
python ./src/main.py \
    --faster-whisper \
    --threads $threads \
    --output-format "$output_format" \
    --output-dir "$output_dir" \
    --language "$language" \
    --model "$model" \
    --audio "$audio_filename" \
    --text "$text_filename"