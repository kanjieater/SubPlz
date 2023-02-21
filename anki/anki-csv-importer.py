#!/usr/bin/env python3

import argparse
import csv
import requests
import os
import tempfile

ANKI_CONNECT_URL = "http://192.168.1.3:8765"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Import a local or remote CSV file into Anki"
    )

    parser.add_argument("-p", "--path", help="the path of the local CSV file")
    parser.add_argument("-u", "--url", help="the URL of the remote CSV file")

    parser.add_argument(
        "-d",
        "--deck",
        help="the name of the deck to import the sheet to",
        required=True,
    )
    parser.add_argument("-n", "--note", help="the note type to import", required=True)

    parser.add_argument(
        "--no-anki-connect",
        help="write notes directly to Anki DB without using AnkiConnect",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--col",
        help="the path to the .anki2 collection (only when using --no-anki-connect)",
    )
    parser.add_argument(
        "--allow-html",
        help="render HTML instead of treating it as plaintext (only when using --no-anki-connect)",
        action="store_true",
    )
    parser.add_argument(
        "--skip-header",
        help="skip first row of CSV (only when using --no-anki-connect)",
        action="store_true",
    )
    parser.add_argument(
        "--expression-index",
        help='mapping of the source csv row number to your note type\'s expression field, "--expression-index 1 --expression-field Expression" would map the 1st aka 0th column from the csv to the expression field named Expression in your note type)',
        action="store",
        type=int,
        required=True
    )
    parser.add_argument(
        "--audio-index",
        help="mapping of the source csv row number to your note type's audio field, --audio-index 3 --expression-field Audio, same as expression",
        action="store",
        type=int,
        required=True
    )
    parser.add_argument(
        "--expression-field",
        help='mapping of the source csv row number to your note type\'s expression field, see as expression',
        action="store",
        type=str,
        required=True
    )
    parser.add_argument(
        "--audio-field",
        help="mapping of the source csv row number to your note type's audio field, see as expression",
        action="store",
        type=str,
        required=True
    )

    return parser.parse_args()


def validate_args(args):
    if args.path and args.url:
        print("[E] Only one of --path and --url can be supplied")
        exit(1)

    if not (args.path or args.url):
        print("[E] You must specify either --path or --url")
        exit(1)

    if args.no_anki_connect:
        if not args.col:
            print("[E] --col is required when using --no-anki-connect")
            exit(1)
    else:
        if args.skip_header:
            print("[E] --skip-header is only supported with --no-anki-connect")
            exit(1)
        elif args.allow_html:
            print(
                "[E] --allow-html is only supported with --no-anki-connect, "
                "when using AnkiConnect HTML is always enabled"
            )
            exit(1)
        elif args.col:
            print("[E] --col is only supported with --no-anki-connect")
            exit(1)


def parse_ac_response(response):
    if len(response) != 2:
        raise Exception("response has an unexpected number of fields")
    if "error" not in response:
        raise Exception("response is missing required error field")
    if "result" not in response:
        raise Exception("response is missing required result field")
    if response["error"] is not None:
        raise Exception(response["error"])
    return response["result"]


def make_ac_request(action, **params):
    return {"action": action, "params": params, "version": 6}


def invoke_ac(action, **params):
    requestJson = make_ac_request(action, **params)
    # try:
    response = requests.post(ANKI_CONNECT_URL, json=requestJson).json()
    # except requests.exceptions.ConnectionError:
    #     print("[E] Failed to connect to AnkiConnect, make sure Anki is running")
    #     exit(1)

    return parse_ac_response(response)


def get_fields(note_type):
    return invoke_ac("modelFieldNames", modelName=note_type)


def map_fields_to_note(row, field_mappings):
    fields = {}
    for field_mapping in field_mappings:
        csv_index, field_name = field_mapping
        fields[field_name] = row[csv_index - 1]

    return fields


def csv_to_ac_notes(csv_path, deck_name, note_type, field_mappings):
    notes = []
    # model_fields = get_fields(note_type)

    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter="\t")
        for i, row in enumerate(reader):
            note = {
                "deckName": deck_name,
                "modelName": note_type,
                "tags": None,
                # 'options': {
                #     # "allowDuplicate": True,
                #     # "duplicateScope": "deck"
                # }
            }
            note["fields"] = map_fields_to_note(row, field_mappings)
            notes.append(note)

    return notes


def get_ac_add_and_update_note_lists(notes):
    result = invoke_ac("canAddNotes", notes=notes)

    notes_to_add = []
    notes_to_update = []
    for i, b in enumerate(result):
        if b:
            notes_to_add.append(notes[i])
        else:
            notes_to_update.append(notes[i])

    return notes_to_add, notes_to_update


def send_to_anki_connect(csv_path, deck_name, note_type, field_mappings):
    notes = csv_to_ac_notes(csv_path, deck_name, note_type, field_mappings)
    invoke_ac("createDeck", deck=deck_name)
    notes_to_add = notes
    print("[+] Adding {} new notes".format(len(notes_to_add)))
    notes_response = invoke_ac("addNotes", notes=notes_to_add)
    successes = [x for x in notes_response if x is not None]
    print("[+] Created {} new notes".format(len(successes)))


def main():
    args = parse_arguments()
    validate_args(args)

    if args.path:
        # Use an existing CSV file. We convert this to an absolute path because
        # CWD might change later
        csv_path = os.path.abspath(args.path)
    else:
        assert False  # Should never reach here

    field_mappings = [(args.expression_index, args.expression_field), (args.audio_index, args.audio_field)]
    send_to_anki_connect(csv_path, args.deck, args.note, field_mappings)


main()
