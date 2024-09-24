import os
import re
from pathlib import Path
from os import path
from os.path import basename, splitext, dirname, isdir, join
from typing import List, Callable
from dataclasses import dataclass
import ffmpeg
from natsort import os_sorted
from glob import glob, escape
from pprint import pformat
from ats.main import (
    TextFile,
    AudioStream,
    TextFile,
    TextParagraph,
    write_srt,
    write_vtt,
)
from subplz.cache import Cache
from subplz.text import split_sentences, split_sentences_from_input, Epub

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
    audio_lang = lang
    if lang == "ja":  # TODO support other languages
        audio_lang = "jpn"
    target_streams = [
        stream
        for stream in audio_streams
        if stream.get("tags", {}).get("language", "").lower() == audio_lang.lower()
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
    cache: Cache

    def transcribe(self, model={}, **kwargs):
        transcription = self.cache.get(os.path.basename(self.path), self.cid)
        if transcription is not None:
            return transcription
        transcription = model.faster_transcribe(self.audio(), self.cn, **kwargs)
        return self.cache.put(os.path.basename(self.path), self.cid, transcription)

    @classmethod
    def from_file(cls, path, cache_inputs={}, whole=False, lang="ja"):
        cache = Cache(**vars(cache_inputs))
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
                    cache=cache,
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
                cache=cache,
            )
            for chapter in info["chapters"]
        ]


@dataclass
class sourceData:
    dirs: List[str]
    audio: List[str]
    output_dir: str
    output_format: str
    overwrite: bool
    rerun: bool
    output_full_paths: List[Path]
    writer: Writer
    streams: List
    lang: str
    text: List[str] = None
    chapters: List = None


def grab_files(folder, types, sort=True):
    files = []
    for t in types:
        pattern = f"{escape(folder)}/{t}"
        files.extend(glob(pattern))
    if sort:
        return os_sorted(files)
    return files


def get_streams(audio, cache_inputs):
    # print("🎧 Loading streams...") #log
    streams = [(basename(f), *AudioSub.from_file(f, cache_inputs)) for f in audio]
    return streams


def convert_sub_format(full_original_path, full_sub_path):
    stream = ffmpeg.input(full_original_path)
    stream = ffmpeg.output(stream, full_sub_path, loglevel="error").global_args(
        "-hide_banner"
    )
    ffmpeg.run(stream, overwrite_output=True)


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
    convert_sub_format(str(file_path), str(srt_path))
    txt_path = remove_timing_and_metadata(srt_path, txt_path)
    srt_path.unlink()
    return str(txt_path)


def get_chapters(text: List[str], lang):
    # print("📖 Finding chapters...") #log
    sub_exts = ["." + extension for extension in SUBTITLE_FORMATS]
    chapters = []
    for file_path in text:
        file_name = basename(file_path)
        file_ext = splitext(file_name)[-1].lower()

        if file_ext == ".epub":
            txt_path = get_tmp_path(
                Path(file_path).parent / f"{Path(file_path).stem}.txt"
            )
            epub = Epub.from_file(file_path)
            chapters.append((txt_path, epub.chapters))
            split_sentences_from_input([p.text() for p in epub.text()], txt_path, lang)
            # chapters.append((txt_path, [TextFile(path=file_path, title=file_name)]))

        elif file_ext in sub_exts:
            try:
                txt_path = normalize_text(file_path)
                split_sentences(txt_path, txt_path, lang)

            except ffmpeg.Error as e:
                print(
                    f"Failed to normalize the subs. We can't process them. Try to get subs from a different source and try again: {e}"
                )
                return []
            chapters.append((txt_path, [TextFile(path=txt_path, title=file_name)]))
        else:
            txt_path = get_tmp_path(
                Path(file_path).parent / f"{Path(file_path).stem}.txt"
            )
            split_sentences(file_path, txt_path, lang)
            chapters.append((txt_path, [TextFile(path=file_path, title=file_name)]))
    return chapters


def get_working_folders(dirs):
    working_folders = []
    for dir in dirs:
        if not isdir(dir):
            raise Exception(f"{dir} is not a valid directory")
        full_folder = join(dir, "")
        working_folder = full_folder
        working_folders.append(working_folder)
    return working_folders


def get_audio(folder):
    audio = grab_files(folder, SUPPORTED_AUDIO_FORMATS)
    return audio


def get_text(folder):
    text = set(grab_files(folder, SUPPORTED_TEXT_FORMATS))
    text = [file_path for file_path in text if not file_path.endswith(".tmp.txt")]
    return os_sorted(text)


def setup_output_dir(output_dir, first_audio=None):
    if not output_dir and first_audio:
        output_dir = Path(first_audio).parent
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_full_paths(audio, output_dir, output_format, lang_ext):
    le = ""
    if lang_ext:
        le = f".{lang_ext}"
    return [Path(output_dir) / f"{Path(a).stem}{le}.{output_format}" for a in audio]


def match_files(audios, texts, folder, rerun):
    if rerun:
        old = get_existing_rerun_files(folder)
        text = grab_files(folder, ["*." + ext for ext in TEXT_FORMATS])
        already_run = os_sorted(list(set(text + old)))
        audios_filtered = audios
        texts_filtered = already_run
    else:
        already_run = get_existing_rerun_files(folder)
        already_run_text_paths = []
        already_run_audio_paths = []
        for ar in already_run:
            arPath = Path(ar)
            removed_second_stem = Path(arPath.stem).stem
            already_run_audio_paths.append(str(arPath.parent / removed_second_stem))
            already_run_text_paths.append(str(arPath.parent / arPath.stem))
        destemed_audio = [
            str(Path(audio).parent / Path(audio).stem) for audio in audios
        ]
        destemed_text = [
            str(Path(text).parent / Path(text).stem) for text in texts
        ]
        audios_unique = list(set(destemed_audio) - set(already_run_audio_paths))
        texts_unique = list(set(destemed_text) - set(already_run_audio_paths) - set(already_run_text_paths))

        texts_filtered = [
            t for t in texts if str(Path(t).parent / Path(t).stem) in texts_unique
        ]
        audios_filtered = [
            a for a in audios if str(Path(a).parent / Path(a).stem) in audios_unique
        ]
    if len(audios_filtered) > 1 and len(texts_filtered) == 1 and len(already_run) == 0:
        print("🤔 Multiple audio files found, but only one text... If you use an epub, each chapter will be mapped to a single audio file")
        return [audios_filtered], [[t] for t in texts_filtered]

    if len(audios_filtered) != len(texts_filtered):
        print(
            "🤔 The number of text files didn't match the number of audio files... Matching them based on sort order. You should probably double-check this."
        )

    return [[a] for a in audios_filtered], [[t] for t in texts_filtered]


def get_sources_from_dirs(input, cache_inputs):
    sources = []
    working_folders = get_working_folders(input.dirs)
    for folder in working_folders:
        audios = get_audio(folder)
        if input.subcommand == "sync":
            texts = get_text(folder)
            a, t = match_files(audios, texts, folder, input.rerun)


            for matched_audio, matched_text in zip(a, t):
                output_full_paths = get_output_full_paths(
                    matched_audio, folder, input.output_format, input.lang_ext
                )
                writer = Writer(input.output_format)

                streams = get_streams(matched_audio, cache_inputs)
                chapters = get_chapters(matched_text, input.lang)
                s = sourceData(
                    dirs=input.dirs,
                    audio=matched_audio,
                    text=matched_text,
                    output_dir=setup_output_dir(folder),
                    output_format=input.output_format,
                    overwrite=input.overwrite,
                    rerun=input.rerun,
                    output_full_paths=output_full_paths,
                    writer=writer,
                    chapters=chapters,
                    streams=streams,
                    lang=input.lang,
                )
                sources.append(s)
        else:
            for matched_audio in audios:
                output_full_paths = get_output_full_paths(
                    [matched_audio], folder, input.output_format, input.lang_ext
                )
                writer = Writer(input.output_format)

                streams = get_streams([matched_audio], cache_inputs)
                s = sourceData(
                    dirs=input.dirs,
                    audio=matched_audio,
                    output_dir=setup_output_dir(folder),
                    output_format=input.output_format,
                    overwrite=input.overwrite,
                    rerun=input.rerun,
                    output_full_paths=output_full_paths,
                    writer=writer,
                    streams=streams,
                    lang=input.lang,
                )
                sources.append(s)
    return sources


def setup_sources(input, cache_inputs) -> List[sourceData]:
    if input.dirs:
        sources = get_sources_from_dirs(input, cache_inputs)
    else:
        if input.subcommand == "sync":
            output_dir = setup_output_dir(input.output_dir, input.audio[0])
            output_full_paths = get_output_full_paths(
                input.audio, output_dir, input.output_format, input.lang_ext
            )
            writer = Writer(input.output_format)
            chapters = get_chapters(input.text, input.lang)
            streams = get_streams(input.audio, cache_inputs)
            sources = [
                sourceData(
                    dirs=[],
                    audio=input.audio,
                    text=input.text,
                    output_dir=output_dir,
                    output_format=input.output_format,
                    overwrite=input.overwrite,
                    rerun=input.rerun,
                    output_full_paths=output_full_paths,
                    writer=writer,
                    streams=streams,
                    chapters=chapters,
                    lang=input.lang,
                )
            ]
        else:
            output_dir = setup_output_dir(input.output_dir, input.audio[0])
            output_full_paths = get_output_full_paths(
                input.audio, output_dir, input.output_format, input.lang_ext
            )
            writer = Writer(input.output_format)
            streams = get_streams(input.audio, cache_inputs)
            sources = [
                sourceData(
                    dirs=[],
                    audio=input.audio,
                    output_dir=output_dir,
                    output_format=input.output_format,
                    overwrite=input.overwrite,
                    rerun=input.rerun,
                    output_full_paths=output_full_paths,
                    writer=writer,
                    streams=streams,
                    lang=input.lang,
                )
            ]
    return sources

def rename_existing_file_to_old(p):
    path_obj = Path(p)
    new_filename = path_obj.with_suffix(".old" + path_obj.suffix)
    path_obj.rename(new_filename)
    return new_filename


def get_sources(input, cache_inputs) -> List[sourceData]:
    sources = setup_sources(input, cache_inputs)
    valid_sources = []
    invalid_sources = []
    for source in sources:
        output_paths = source.output_full_paths
        is_valid = True
        for op in output_paths:
            old_file = get_rerun_file_path(op)
            if not source.overwrite and op.exists():
                print(f"🤔 SubPlz file '{op.name}' already exists, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break
            if old_file.exists() and op.exists() and not source.rerun:
                print(
                    f"🤔 {old_file.name} already exists but you don't want it overwritten, skipping. If you do, add --rerun"
                )
                invalid_sources.append(source)
                is_valid = False
                break
            if not source.audio:
                print(f"❗ {op.name}'s audio is missing, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break
            if not source.text and input.subcommand == "sync":
                print(f"❗ {source.audio}'s text is missing, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break
            if not source.chapters and input.subcommand == "sync":
                print(f"❗ {source.text}'s couldn't be parsed, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break
            if op.exists() and not old_file.exists() and input.subcommand == "sync":
                rename_existing_file_to_old(op)

        if is_valid:
            valid_sources.append(source)

    if input.subcommand == "sync":
        for source in valid_sources:
            print(f"🎧 {pformat(source.audio)}' ⟹ 📖 {pformat(source.text)}...")
        cleanup(invalid_sources)
    return valid_sources


def cleanup(sources: List[sourceData]):
    for source in sources:
        for file in source.text:
            tmp_file = get_tmp_path(file).with_suffix(".txt")
            tmp_file_path = Path(tmp_file)
            if tmp_file_path.exists():
                tmp_file_path.unlink()


def get_existing_rerun_files(dir: str) -> List[str]:
    old = grab_files(dir, ["*.old." + ext for ext in SUBTITLE_FORMATS])
    return old


def get_rerun_file_path(output_path: Path) -> Path:
    cache_file = output_path.parent / f"{output_path.stem}.old{output_path.suffix}"
    return cache_file


def rename_old_subs(source: sourceData):
    subs = []
    for sub in source.text:
        if (
            Path(sub).suffix[1:] in SUBTITLE_FORMATS
            and ".old" not in Path(Path(sub).stem).suffix
        ):
            subs.append(sub)
    remaining_subs = set(subs) - set([str(p) for p in source.output_full_paths])
    for i, sub in enumerate(remaining_subs):
        sub_path = Path(sub)

        if len(source.text) == len(source.audio):
            new_filename = Path(source.audio[i]).with_suffix(".old" + sub_path.suffix)
        else:
            new_filename = sub_path.with_suffix(".old" + sub_path.suffix)
        sub_path.rename(new_filename)


def post_process(sources: List[sourceData], subcommand, alass):
    if subcommand == "sync":
        cleanup(sources)
    complete_success = True
    sorted_sources = sorted(
        sources, key=lambda source: source.writer.written, reverse=True
    )
    for source in sorted_sources:
        if source.writer.written:
            output_paths = [str(o) for o in source.output_full_paths]
            if subcommand == "sync":
                rename_old_subs(source)
            print(f"🙌 Successfully wrote '{', '.join(output_paths)}'")
        elif alass:
            print(f"Alass completed for '{source.text}'")
        else:
            complete_success = False
            print(f"❗ No text matched for '{source.text}'")

    if not sources and not alass:
        print(
            """😐 We didn't do anything. This may or may not be intentional. If this was unintentional, check if you had a .old file preventing rerun"""
        )
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
