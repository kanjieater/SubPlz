#!/bin/bash

# Define variables
audio_filename="/sync/変な家/変な家.m4b"
text_filename="/sync/変な家/変な家.epub"
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
docker run -d \
  --name=faster-whisper \
  -e PUID=1000 \
  -e PGID=1000 \
  -e TZ=Etc/UTC \
  -e WHISPER_MODEL=tiny-int8 \
  -e WHISPER_BEAM=1 `#optional` \
  -e WHISPER_LANG=en `#optional` \
  -p 10300:10300 \
  -v /path/to/data:/config \
  --restart unless-stopped \
  lscr.io/linuxserver/faster-whisper:latest