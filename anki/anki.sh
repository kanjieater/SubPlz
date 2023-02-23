#!/bin/bash
set -e

FOLDER="${1:-pwd}"
NAME="$(basename "$FOLDER" )"
ANKICONNECTURL="${2:-$ANKICONNECT}"
MAPPING="${3:-"./anki/mapping.json"}"
echo "Connecting to AnkiConnect via $ANKICONNECTURL"
echo "Mapping: $MAPPING";
echo "TSV: $FOLDER/srs_export/$NAME.tsv";

python -m subs2cia srs -i "$FOLDER/$NAME.m4b" "$FOLDER/$NAME.srt" -d "$FOLDER/srs_export" -p 100
python ./anki/anki-importer.py --anki-connect-url="$ANKICONNECTURL" --path "$FOLDER/srs_export/$NAME.tsv" --mapping "$MAPPING" --name "$NAME"