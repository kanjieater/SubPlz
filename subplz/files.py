import os
import re
from pathlib import Path
from os import path
from os.path import basename, splitext, dirname, isdir, join
from typing import List, Callable
from dataclasses import dataclass
from natsort import os_sorted
from glob import glob, escape
from pprint import pformat
from ats.main import (
    TextFile,
    Epub,
    AudioStream,
    TextFile,
    TextParagraph,
    write_srt,
    write_vtt,
)
from functools import partial
import ffmpeg

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
    "*." + extension for extension in text_formats + subtitle_formats
]


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


def get_video_duration(stream, file_path):
    try:
        if "duration" in stream:
            duration = float(stream["duration"])
        else:
            probe = ffmpeg.probe(file_path)
            duration = float(probe["format"]["duration"])
        print(f"Duration: {duration}")
        return duration
    except ffmpeg.Error as e:
        error_message = f"ffmpeg error: {e.stderr.decode('utf-8')}"
        raise RuntimeError(error_message)
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        raise RuntimeError(error_message)


def get_matching_audio_stream(streams, lang):
    audio_streams = [
        stream for stream in streams if stream.get("codec_type", None) == "audio"
    ]
    target_streams = [
        stream
        for stream in audio_streams
        if stream.get("tags", {}).get("language", None) == lang
    ]
    return next((stream for stream in target_streams + audio_streams), None)


@dataclass(eq=True, frozen=True)
class AudioSub(AudioStream):
    stream: ffmpeg.Stream
    path: Path
    duration: float
    cn: str
    cid: int

    @classmethod
    def from_file(cls, path, whole=False, lang="ja"):
        try:
            info = ffmpeg.probe(path, show_chapters=None)
        except ffmpeg.Error as e:
            print(e.stderr.decode("utf8"))
            exit(1)

        title = info.get("format", {}).get("tags", {}).get("title", basename(path))

        if whole or "chapters" not in info or len(info["chapters"]) == 0:
            stream = get_matching_audio_stream(info["streams"], lang)
            duration = get_video_duration(stream, path)
            return title, [
                cls(
                    stream=ffmpeg.input(path),
                    duration=duration,
                    path=path,
                    cn=title,
                    cid=0,
                )
            ]

        return title, [
            cls(
                stream=ffmpeg.input(
                    path, ss=float(chapter["start_time"]), to=float(chapter["end_time"])
                ),
                duration=float(chapter["end_time"]) - float(chapter["start_time"]),
                path=path,
                cn=chapter.get("tags", {}).get("title", ""),
                cid=chapter["id"],
            )
            for chapter in info["chapters"]
        ]


# @dataclass(eq=True, frozen=True)
# class TextSub:
#     path: str
#     title: str

#     def name(self):
#         return self.title

#     def text(self, *args, **kwargs):
#         # Read the file and split it into lines
#         lines = Path(self.path).read_text().split('\n')
#         paragraphs = []

#         for i, line in enumerate(lines):
#             stripped_line = line.strip()
#             if stripped_line and not self._is_timing_line(stripped_line):
#                 paragraphs.append(TextParagraph(path=self.path, idx=i, content=stripped_line, references=[]))

#         return paragraphs

#     def _is_timing_line(self, line: str) -> bool:
#         # Regex pattern to match timing lines in SRT format
#         timing_pattern = r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$'
#         return re.match(timing_pattern, line) is not None


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
    streams = [(basename(f), *AudioSub.from_file(f)) for f in audio]
    return streams

def convert_to_srt(file_path):
    filename, ext = path.splitext(file_path)
    stream = ffmpeg.input(file_path)
    stream = ffmpeg.output(
        stream,
        path.join(path.dirname(file_path), f"{filename}.srt"),
        vn=None,
        loglevel="error",
    ).global_args("-hide_banner")
    return ffmpeg.run(stream, overwrite_output=True)

def convert_to_txt(file_path):
    return


def normalize_text(file_path):
    convert_to_srt(file_path)
    srt_path = file_path.replace(".srt", ".txt")
    convert_to_txt(srt_path)
    txt_path = file_path.replace(".srt", ".txt")
    return txt_path


def get_chapters(text: List[str]):
    print("Finding chapters...")
    chapters = []
    for file_path in text:
        file_name = basename(file_path)
        file_ext = splitext(file_name)[-1].lower()

        if file_ext == ".epub":
            chapters.append((file_name, Epub.from_file(file_path)))
        elif file_ext == ".ass":
            txt_path = normalize_text(file_path)
            chapters.append((file_name, [TextFile(path=txt_path, title=file_name)]))
        else:
            chapters.append((file_name, [TextFile(path=file_path, title=file_name)]))
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
            writer=writer,
        )
        sources.append(s)
    return sources


def get_output_full_paths(audio, output_dir, output_format):
    return [Path(output_dir) / f"{Path(a).stem}.{output_format}" for a in audio]


def write_sub(output_format, segments, output_full_path):
    with output_full_path.open("w", encoding="utf8") as o:
        print(f"Writing to '{output_full_path}'")
        if output_format == "srt":
            return write_srt(segments, o)
        elif output_format == "vtt":
            return write_vtt(segments, o)


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
                continue
            if not source.audio:
                print(f"{fp.name}'s audio is missing, skipping.")
                valid_sources.pop(-1)
                continue
            if not source.text:
                print(f"{source.audio}'s text is missing, skipping.")
                valid_sources.pop(-1)
                continue

    for source in valid_sources:
        print(f"'{pformat(source.audio)}' will be matched to {pformat(source.text)}...")
    return valid_sources
