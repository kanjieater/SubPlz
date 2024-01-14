import argparse
import sys, os
import stable_whisper
import ffmpeg
import multiprocessing
import traceback
from os import path
from pathlib import Path
from datetime import datetime, timedelta
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from pprint import pprint
from utils import read_vtt, write_sub, grab_files, audio_formats, video_formats, subtitle_formats, get_mapping
from run import get_working_folders, generate_transcript_from_audio, get_model




SUPPORTED_FORMATS = ['*.' + extension for extension in video_formats + audio_formats]

def match_subs(files, model):
    for file in files:
        try:
            ext = 'srt'
            lang_code = '.ja'
            sub_path = str(Path(file).with_suffix(lang_code))
            generate_transcript_from_audio(file, sub_path, model, ext)
        except Exception as err:
            tb = traceback.format_exc()
            pprint(tb)
            failures.append({"working_folder": file, "err": err})
    return failures


def get_mapping_config():
    script_path = Path(__file__).resolve().parent
    json_file_name = "sub.json"
    json_file_path = script_path / json_file_name
    return get_mapping(str(json_file_path))


def get_matching_dirs(config):
    content_dirs = config['content_dirs']
    sub_dir = config['sub_dir']
    blacklist_dirs = config['blacklist_dirs']

    sub_entries = [entry.name for entry in Path(sub_dir).iterdir() if entry.is_dir()]
    matching_dirs = []

    for content_dir in content_dirs:
        content_path = Path(content_dir)
        content_entries = [entry.name for entry in content_path.iterdir() if entry.is_dir()]
        for content_entry in content_entries:
            if content_entry in sub_entries and content_entry not in blacklist_dirs:
                matching_dirs.append(content_entry)

    return matching_dirs

def get_folders_with_matching_subs(content_names):
    for name in content_names:
        pass

if __name__ == "__main__":

    # args = parser.parse_args()
    config = get_mapping_config()
    content_names = get_matching_dirs(config)
    valid_content_folders = get_folders_with_matching_subs(content_names, config.content_dirs, config.sub_dir)

    global model
    model = False  # global model preserved between files
    model = get_model('large-v2')
    successes = []
    failures = []
    for working_folder in working_folders:
        # try:
        print(working_folder)
        audio_files = grab_files(working_folder, SUPPORTED_FORMATS)
        failures.extend(match_subs(audio_files, model))
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
            pprint(failure['working_folder'])
            pprint(failure['err'])
