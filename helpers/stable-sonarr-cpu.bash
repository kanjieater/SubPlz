find . -type f -name "*.mkv" -exec sh -c '
  if [ ! -f "${1%.mkv}.ru.json" ] &&
    ! ([ -f "${1%.mkv}.ru.srt" ]) &&
    ! ([ -f "${1%.mkv}.jp.srt" ] || [ -f "${1%.mkv}.ja.srt" ] ||
        [ -f "${1%.mkv}.jp.ass" ] || [ -f "${1%.mkv}.ja.ass" ]); then
      if [ "$(date +%s -r "$1")" -le "$(($(date +%s) - 300))" ]; then
        echo "stable-ts \"$1\" -o \"${1%.mkv}.ru.json\" --model large-v2 --refine --language ja --fp16=False --denoiser=demucs --device=cpu"
        stable-ts "$1" -o "${1%.mkv}.ru.json" --model large-v2 --refine --language ja --fp16=False --denoiser=demucs --device=cpu
      fi
  fi
' _ {} \; &&
find . -type f -name "*.ru.json" -exec sh -c '
  if [ ! -f "${1%.ru.json}.ru.srt" ]; then
    echo "stable-ts \"$1\" -o \"${1%.ru.json}.ru.srt\" --word_level false"
    stable-ts "$1" -o "${1%.ru.json}.ru.srt" --word_level false
  fi
' _ {} \;
