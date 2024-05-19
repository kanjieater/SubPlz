import os
from pathlib import Path
from os import path
from os.path import basename, splitext, dirname, isdir, join
from typing import List, Callable
from dataclasses import dataclass
from natsort import os_sorted
from glob import glob, escape
from pprint import pformat
from ats.main import TextFile, Epub, AudioStream, write_srt, write_vtt
from functools import partial


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
text_formats = ["epub", "txt"]

SUPPORTED_AUDIO_FORMATS = [
    "*." + extension for extension in video_formats + audio_formats
]
SUPPORTED_TEXT_FORMATS = ["*." + extension for extension in text_formats]


@dataclass
class sourceData:
    dirs: List[str]
    audio: List[str]
    text: List[str]
    output_dir: str
    output_format: str
    overwrite: bool
    output_full_paths: str
    writer: Callable[[str, str], None]


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
    streams = [(basename(f), *AudioStream.from_file(f)) for f in audio]
    return streams


def get_chapters(text):
    print("Finding chapters...")
    chapters = [
        (
            (basename(i), Epub.from_file(i))
            if i.split(".")[-1] == "epub"
            else (basename(i), [TextFile(path=i, title=basename(i))])
        )
        for i in text
    ]
    return chapters


def get_working_folders(dirs):
    working_folders = []
    for dir in dirs:
        if not isdir(dir):
            raise Exception(f"{dir} is not a valid directory")
        full_folder = join(dir, "")
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
        audio = get_audio(folder)
        output_full_paths = get_output_full_paths(audio, folder, input.output_format)
        writer = get_writer(input.output_format)
        s = sourceData(
            dirs=input.dirs,
            audio=audio,
            text=get_text(folder),
            output_dir=setup_output_dir(folder),
            output_format=input.output_format,
            overwrite=input.overwrite,
            output_full_paths=output_full_paths,
            writer=writer
        )
        sources.append(s)
    return sources


def get_output_full_paths(audio, output_dir, output_format):
    return [
        Path(output_dir) / f"{Path(a).stem}.{output_format}"
        for a in audio
    ]

def write_sub(output_format, segments, output_full_path):
    with output_full_path.open("w", encoding="utf8") as o:
        if output_format == "srt":
            return write_srt(segments, o)
        elif output_format == "vtt":
            return write_vtt(segments, o)
        print(f"Output to '{output_full_path}'")


def get_writer(output_format):
    return partial(write_sub, output_format)


def setup_sources(input):
    if input.dirs:
        sources = get_sources_from_dirs(input)
    else:
        output_full_paths = get_output_full_paths(
            input.audio, input.output_dir, input.output_format
        )
        writer = get_writer(input.output_format)
        sources = [
            sourceData(
                dirs=[],
                audio=input.audio,
                text=input.text,
                output_dir=setup_output_dir(input.output_dir),
                output_format=input.output_format,
                overwrite=input.overwrite,
                output_full_paths=output_full_paths,
                writer=writer,
            )
        ]
    return sources


def get_sources(input):
    sources = setup_sources(input)
    valid_sources = []
    for source in sources:
        paths = source.output_full_paths
        for fp in paths:
            valid_sources.append(source)
            if not source.overwrite and fp.exists():
                print(f"{fp.name} already exists, skipping.")
                valid_sources.pop(-1)
    for source in valid_sources:
        print(f"'{pformat(source.audio)}' will be matched to {pformat(source.text)}...")
    return valid_sources
