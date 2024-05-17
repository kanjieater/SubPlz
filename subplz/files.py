import os
from pathlib import Path
from os import path
from typing import List
from dataclasses import dataclass
from natsort import os_sorted
from glob import glob, escape
from pprint import pformat
from ats.main import TextFile, Epub, AudioStream

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
text_formats = ["epub", "text"]

SUPPORTED_AUDIO_FORMATS = [
    "*." + extension for extension in video_formats + audio_formats
]
SUPPORTED_TEXT_FORMATS = [
    "*." + extension for extension in text_formats
]


@dataclass
class sourceData:
    dirs: List[str]
    audio: List[str]
    text: List[str]
    output_dir: str
    output_format: str
    overwrite: bool


def grab_files(folder, types, sort=True):
    files = []
    for type in types:
        pattern = f"{escape(folder)}/{type}"
        files.extend(glob(pattern))
    if sort:
        return os_sorted(files)
    return files


def get_streams(audio):
    print("Loading streams...")
    streams = [(os.path.basename(f), *AudioStream.from_file(f)) for f in audio]
    return streams


def get_chapters(text):
    print("Finding chunks...")
    chapters = [
        (
            (os.path.basename(i), Epub.from_file(i))
            if i.split(".")[-1] == "epub"
            else (os.path.basename(i), [TextFile(path=i, title=os.path.basename(i))])
        )
        for i in text
    ]
    return chapters


def get_content_name(working_folder):
    folder_name = path.dirname(working_folder)
    content_name = path.basename(folder_name)
    return content_name


def get_working_folders(dirs):
    working_folders = []
    for dir in dirs:
        if not path.isdir(dir):
            raise Exception(f"{dir} is not a valid directory")
        full_folder = path.join(dir, "")
        # content_name = get_content_name(dir)
        # split_folder = path.join(full_folder, f"{content_name}_splitted")

        # if path.exists(split_folder) and path.isdir(split_folder):
        #     working_folder = split_folder
        #     print(
        #         f"Warning: Using split files causes a fixed delay for every split file. This is a known bug. Use the single file method instead"
        #     )
        # else:
        working_folder = full_folder
        working_folders.append(working_folder)
    return working_folders


def get_audio(folder):
    audio = grab_files(folder, SUPPORTED_AUDIO_FORMATS)
    return audio


def get_text(folder):
    text = grab_files(folder, SUPPORTED_TEXT_FORMATS)
    return text

def get_output_dir(folder, output_format):
    pass

def setup_output_dir(output_dir):
    output_dir = Path(o) if (o := output_dir) else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_sources_from_dirs(input):
    sources = []
    working_folders = get_working_folders(input.dirs)
    for folder in working_folders:
        s = sourceData(
            dirs=input.dirs,
            audio=get_audio(folder),
            text=get_text(folder),
            output_dir=setup_output_dir(folder),
            output_format=input.output_format,
            overwrite=input.overwrite,
        )
        sources.append(s)
    return sources

def get_sources(input):
    sources = [sourceData(
            dirs=[],
            audio=input.audio,
            text=input.text,
            output_dir=setup_output_dir(input.output_dir),
            output_format=input.output_format,
            overwrite=input.overwrite,
        )]
    if input.dirs:
        sources = get_sources_from_dirs(input)
    for source in sources:
        print(f"'{pformat(source.audio)} will be matched to {pformat(source.text)}...")
    return sources
