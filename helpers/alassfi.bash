find . \( -name "*.fi.srt" -o -name "*.fi.hi.srt" \) | while read fi_srt; do
    # Check if file ends with .fi.hi.srt, if not, it's a .fi.srt
    if [[ "$fi_srt" == *.fi.hi.srt ]]; then
        pt_srt="${fi_srt/.fi.hi.srt/.pt.srt}"
        es_srt="${fi_srt/.fi.hi.srt/.es.srt}"
    else
        pt_srt="${fi_srt/.fi.srt/.pt.srt}"
        es_srt="${fi_srt/.fi.srt/.es.srt}"
    fi

    # Check if the corresponding .pt.srt file exists
    if [[ -f "$pt_srt" ]]; then
        # Check if the output file already exists
        if [[ -f "$es_srt" ]]; then
            echo "Skipping: Output file $es_srt already exists."
        else
            # Run alass-cli to sync subtitles
            /home/anonymous/.cargo/bin/alass-cli "$pt_srt" "$fi_srt" "$es_srt"
            echo "Processed: $fi_srt with $pt_srt -> $es_srt"
        fi
    else
        echo "No corresponding Portuguese subtitle for $fi_srt"
    fi
done
