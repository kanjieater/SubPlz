import argparse
import sys, os
import stable_whisper
import ffmpeg
import multiprocessing

# from subsai import SubsAI
import traceback
from os import path
from pathlib import Path
from datetime import datetime, timedelta
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from pprint import pprint
from utils import read_vtt, write_sub, grab_files

# from split_sentences import split_sentences
from run import get_working_folders, generate_transcript_from_audio, get_model

audio_formats = [
    "aac",
    "ac3",
    "alac",
    "ape",
    "flac",
    "mp3",
    "m4a",
    "ogg",
    "opus",
    "wav",
    "m4b",
]
video_formats = ["3g2", "3gp", "avi", "flv", "m4v", "mkv", "mov", "mp4", "mpeg", "webm"]
subtitle_formats = ["ass", "srt", "vtt"]

SUPPORTED_FORMATS = ["*." + extension for extension in video_formats + audio_formats]


def generate_subs(files, model):
    for file in files:
        try:
            ext = "srt"
            lang_code = ".ja"
            sub_path = str(Path(file).with_suffix(lang_code))
            generate_transcript_from_audio(file, sub_path, model, ext)
        except Exception as err:
            tb = traceback.format_exc()
            pprint(tb)
            failures.append({"working_folder": file, "err": err})
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio for files")
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

    parser.add_argument(
        "--use-filtered-cache",
        help="Uses cached filtered files and skips cleanup. Skips making the filtered audio files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use-transcript-cache",
        help="Uses cached transcript files and skips cleanup. Skips the whisper step.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--align",
        help="Align existing subs",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-sd",
        "--subs_dir",
        dest="subs_dir",
        default=None,
        required=False,
        type=str,
        nargs="+",
        help="List of folders that have subs, in the same order as dirs",
    )

    args = parser.parse_args()
    working_folders = get_working_folders(args.dirs)
    global model
    model = False  # global model preserved between files
    model = get_model("large-v2")
    successes = []
    failures = []
    for working_folder in working_folders:
        # try:
        print(working_folder)
        audio_files = grab_files(working_folder, SUPPORTED_FORMATS)
        failures.extend(generate_subs(audio_files, model))
        successes.append(working_folder)
        # except Exception as err:
        #     pprint(err)
        #     failures.append({"working_folder": working_folder, "err": err})

    if len(successes) > 0:
        print(f"The following {len(successes)} succeeded:")
        pprint(successes)
    if len(failures) > 0:
        print(f"The following {len(failures)} failed:")
        for failure in failures:
            pprint(failure["working_folder"])
            pprint(failure["err"])
