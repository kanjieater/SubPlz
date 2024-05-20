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

AUDIO_FORMATS = [
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
VIDEO_FORMATS = ["3g2", "3gp", "avi", "flv", "m4v", "mkv", "mov", "mp4", "mpeg", "webm"]
SUBTITLE_FORMATS = ["ass", "srt", "vtt"]
TEXT_FORMATS = ["epub", "txt"]

SUPPORTED_AUDIO_FORMATS = [
    "*." + extension for extension in VIDEO_FORMATS + AUDIO_FORMATS
]
SUPPORTED_TEXT_FORMATS = [
    "*." + extension for extension in TEXT_FORMATS + SUBTITLE_FORMATS
]
SUPPORTED_SUBTITLE_FORMATS = ["*." + extension for extension in SUBTITLE_FORMATS]


def get_video_duration(stream, file_path):
    try:
        if "duration" in stream:
            duration = float(stream["duration"])
        else:
            probe = ffmpeg.probe(file_path)
            duration = float(probe["format"]["duration"])
        # print(f"Duration: {duration}") #log
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


class Writer:
    def __init__(self, output_format="srt"):
        self.written = False
        self.output_format = output_format

    def write_sub(self, segments, output_full_path):
        self.output_format = self.output_format
        with output_full_path.open("w", encoding="utf8") as o:
            # print(f"Writing to '{output_full_path}'") # log-
            if self.output_format == "srt":
                self.written = True
                return write_srt(segments, o)
            elif self.output_format == "vtt":
                self.written = True
                return write_vtt(segments, o)


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
            raise RuntimeError(f"'{path}' ffmpeg error: {e.stderr.decode('utf-8')}")

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


@dataclass
class sourceData:
    dirs: List[str]
    audio: List[str]
    text: List[str]
    output_dir: str
    output_format: str
    overwrite: bool
    output_full_paths: List[Path]
    writer: Writer
    chapters: List
    streams: List


def grab_files(folder, types, sort=True):
    files = []
    for type in types:
        pattern = f"{escape(folder)}/{type}"
        files.extend(glob(pattern))
    if sort:
        return os_sorted(files)
    return files


def get_streams(audio):
    # print("🎧 Loading streams...") #log
    streams = [(basename(f), *AudioSub.from_file(f)) for f in audio]
    return streams


def convert_to_srt(file_path, output_path):

    stream = ffmpeg.input(str(file_path))
    stream = ffmpeg.output(
        stream,
        str(output_path),
        vn=None,
        loglevel="error",
    ).global_args("-hide_banner")

    return ffmpeg.run(stream, overwrite_output=True)


def remove_timing_and_metadata(srt_path, txt_path):
    with (
        open(srt_path, "r", encoding="utf-8") as srt_file,
        open(txt_path, "w", encoding="utf-8") as txt_file,
    ):
        for line in srt_file:
            # Skip lines that contain only numbers or '-->'
            if line.strip() and not line.strip().isdigit() and "-->" not in line:
                clean_line = re.sub(r"<[^>]+>", "", line.strip())  # Remove HTML tags
                clean_line = re.sub(r"{[^}]+}", "", clean_line)  # Remove Aegisub tags
                clean_line = re.sub(r"m\s\d+\s\d+\s.+?$", "", clean_line)

                txt_file.write(clean_line + "\n")
    return str(txt_path)


def get_tmp_path(file_path):
    file_path = Path(file_path)
    filename = file_path.stem
    return file_path.parent / f"{filename}.tmp{file_path.suffix}"


def normalize_text(file_path):
    file_path = Path(file_path)
    filename = file_path.stem
    srt_path = get_tmp_path(file_path.parent / f"{filename}.srt")
    txt_path = get_tmp_path(file_path.parent / f"{filename}.txt")
    convert_to_srt(file_path, srt_path)
    txt_path = remove_timing_and_metadata(srt_path, txt_path)
    srt_path.unlink()
    return str(txt_path)


def get_chapters(text: List[str]):
    # print("📖 Finding chapters...") #log
    sub_exts = ["." + extension for extension in SUBTITLE_FORMATS]
    chapters = []
    for file_path in text:
        file_name = basename(file_path)
        file_ext = splitext(file_name)[-1].lower()

        if file_ext == ".epub":
            chapters.append((file_name, Epub.from_file(file_path)))
        elif file_ext in sub_exts:
            try:
                txt_path = normalize_text(file_path)
            except ffmpeg.Error as e:
                print(
                    f"Failed to normalize the subs. Can't process them, get's subs from a different source and try again: {e}"
                )
                return []
            chapters.append((txt_path, [TextFile(path=txt_path, title=file_name)]))
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
    text = set(grab_files(folder, SUPPORTED_TEXT_FORMATS))
    text = [file_path for file_path in text if not file_path.endswith('.tmp.txt')]
    return os_sorted(text)


def get_output_dir(folder, output_format):
    pass


def setup_output_dir(output_dir):
    output_dir = Path(o) if (o := output_dir) else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_full_paths(audio, output_dir, output_format):
    return [Path(output_dir) / f"{Path(a).stem}.{output_format}" for a in audio]


def match_files(audios, texts):
    if len(audios) > 1 and len(texts) == 1:
        print("🤔 Multiple audio files found, but only one text...")
        return [audios], [[t] for t in texts]

    if len(audios) != len(texts):
        print(
            "🤔 The number of text files didn't match the number of audio files... Matching them based on sort order. You should probably double-check this."
        )
    return [[a] for a in audios], [[t] for t in texts]


def get_sources_from_dirs(input):
    sources = []
    working_folders = get_working_folders(input.dirs)
    for folder in working_folders:
        audios = get_audio(folder)
        texts = get_text(folder)
        a, t = match_files(audios, texts)
        for matched_audio, matched_text in zip(a, t):
            output_full_paths = get_output_full_paths(
                matched_audio, folder, input.output_format
            )
            writer = Writer(input.output_format)

            streams = get_streams(matched_audio)
            chapters = get_chapters(matched_text)
            s = sourceData(
                dirs=input.dirs,
                audio=matched_audio,
                text=matched_text,
                output_dir=setup_output_dir(folder),
                output_format=input.output_format,
                overwrite=input.overwrite,
                output_full_paths=output_full_paths,
                writer=writer,
                chapters=chapters,
                streams=streams,
            )
            sources.append(s)
    return sources


def setup_sources(input) -> List[sourceData]:
    if input.dirs:
        sources = get_sources_from_dirs(input)
    else:
        output_full_paths = get_output_full_paths(
            input.audio, input.output_dir, input.output_format
        )
        writer = Writer(input.output_format)
        chapters = get_chapters(input.text)
        streams = get_streams(input.audio)
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
                streams=streams,
                chapters=chapters,
            )
        ]
    return sources


def get_sources(input):
    sources = setup_sources(input)
    valid_sources = []
    invalid_sources = []
    for source in sources:
        paths = source.output_full_paths
        is_valid = True
        for fp in paths:
            cache_file = get_sub_cache_path(fp)
            if not source.overwrite and fp.exists():
                print(f"🤔 SubPlz file '{fp.name}' already exists, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break
            if cache_file.exists() and fp.exists():
                print(
                    f"🤔 {cache_file.name} already exists but you don't want it overwritten, skipping."
                )
                invalid_sources.append(source)
                is_valid = False
                break
            if not source.audio:
                print(f"❗ {fp.name}'s audio is missing, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break
            if not source.text:
                print(f"❗ {source.audio}'s text is missing, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break
            if not source.chapters:
                print(f"❗ {source.text}'s couldn't be parsed, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break

        if is_valid:
            valid_sources.append(source)

    for source in valid_sources:
        print(f"🎧 {pformat(source.audio)}' ⟹ 📖 {pformat(source.text)}...")
    cleanup(invalid_sources)
    return valid_sources


def cleanup(sources: List[sourceData]):
    for source in sources:
        for file in source.text:
            tmp_file = get_tmp_path(file).with_suffix('.txt')
            tmp_file_path = Path(tmp_file)
            if tmp_file_path.exists():
                tmp_file_path.unlink()


def get_sub_cache_path(output_path: Path) -> str:
    cache_file = output_path.parent / f"{output_path.stem}{output_path.suffix}.subplz"
    return cache_file



def write_sub_cache(source: sourceData):
    for file in source.output_full_paths:
        cache_file = get_sub_cache_path(file)
        cache_file.touch()  # Create an empty file
        # print(f"Created cache file: {cache_file}") #log


def rename_old_subs(source: sourceData):
    subs = []
    for sub in source.text:
        if Path(sub).suffix[1:] in SUBTITLE_FORMATS:
            subs.append(sub)
    remaining_subs = set(subs) - set(source.output_full_paths)

    for sub in remaining_subs:
        sub_path = Path(sub)
        new_filename = sub_path.with_suffix(sub_path.suffix + ".old")
        sub_path.rename(new_filename)


def post_process(sources: List[sourceData]):
    cleanup(sources)
    complete_success = True
    sorted_sources = sorted(
        sources, key=lambda source: source.writer.written, reverse=True
    )
    for source in sorted_sources:
        if source.writer.written:
            output_paths = [str(o) for o in source.output_full_paths]
            rename_old_subs(source)
            write_sub_cache(source)
            print(f"🙌 Successfully wrote '{', '.join(output_paths)}'")
        else:
            complete_success = False
            print(f"❗ No text matched for '{source.text}'")

    if not sources:
        print("""😐 We didn't do anything. This may or may not be intentional""")
    elif complete_success:
        print("🎉 Everything went great!")
    else:
        print(
            """😭 At least one of the files failed to sync.
            Possible reasons:
            1. The audio didn't match the text.
            2. The audio and text file matching might not have been ordered correctly.
            3. It could be cached - You could try running with `--overwrite-cache` if you've renamed another file to the exact same file path that you've run with the tool before.
              """
        )
