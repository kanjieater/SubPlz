find . -name "*.mkv" -exec bash -c '
  for file do
    file_dir=$(dirname "$file")
    mkv_file=$(basename "$file" .mkv)
    srt_file="${file_dir}/${mkv_file}.pt.srt"

    if [[ ! -f "$srt_file" ]]; then
      ffmpeg -i "$file" -map 0:s:0 -c:s srt "$srt_file"
    else
      echo "Skipping ${mke_full}.mkv. ${mkv_file}.pt.srt already exists."
    fi
  done
' _ {} +
