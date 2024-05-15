import argparse

from ats.main import match_start
from file import get_working_folders
from src.transcribe import transcribe
from src.sync import sync


def setup_cli():
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    parser.add_argument(
        "-d",
        "--dirs",
        dest="dirs",
        default=None,
        required=True,
        type=str,
        nargs="+",
        help="List of folders to run generate subs for",
    )

    args = parser.parse_args()
    return args


def execute_on_inputs():
    args = setup_cli()
    print(args)
    working_folders = get_working_folders(args.dirs)
    print(working_folders)
    audio_batches = transcribe(streams, model, cache, temperature, threads, args)
    sync(streams, chapters, cache)


if __name__ == "__main__":
    execute_on_inputs()



# if __name__ == "__main__":
#     streams =[]
#     chapters=[]
#     cache=[]
#     print('Fuzzy matching chapters...')
#     ats, sta = match_start(streams, chapters, cache)
#     # audio_batches = expand_matches(streams, chapters, ats, sta)
#     # print_batches(audio_batches)