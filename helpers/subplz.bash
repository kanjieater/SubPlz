find . -type f -name '*.mkv' -exec sh -c '
for mkvfile; do
    outputfile="${mkvfile%.mkv}.ar.srt"
    failsubfile="${mkvfile%.mkv}.failsub"
    if [ -f "$outputfile" ]; then
        echo "Skipping $mkvfile as $outputfile already exists."
        continue
    fi
    if [ -f "$failsubfile" ]; then
        echo "Skipping $mkvfile as $failsubfile already exists."
        continue
    fi
    # Determine the correct subtitle file
    subtitlefile=""
    if [ -f "${mkvfile%.mkv}.fi.srt" ]; then
        subtitlefile="${mkvfile%.mkv}.fi.srt"
    elif [ -f "${mkvfile%.mkv}.hi.fi.srt" ]; then
        subtitlefile="${mkvfile%.mkv}.hi.fi.srt"
    else
        echo "No subtitle file found for $mkvfile"
        continue
    fi
    subplz sync --audio "$mkvfile" --text "$subtitlefile" --language ja --model "large-v3" --output-format srt --progress --overwrite-cache
    mv "${mkvfile%.mkv}.srt" "$outputfile"
    if [ -f "${mkvfile%.mkv}.old.srt" ]; then
        mv "${mkvfile%.mkv}.old.srt" "$subtitlefile"
    fi
    rm -rf SyncCache
    # Check if the .ar.srt file exists, and create a .failsub file if it doesn'\''t
    if [ ! -f "$outputfile" ]; then
        touch "$failsubfile"
    fi
done
' sh {} +
