import os
from collections import defaultdict
from pathlib import Path
from os.path import basename, splitext, isdir, join
from typing import List
from dataclasses import dataclass
import ffmpeg
from natsort import os_sorted
import stanza

from pprint import pformat
from ats.main import (
    AudioStream,
    TextFile,
    write_srt,
    write_vtt,
)
from .logger import logger
from .cache import Cache
from .audio import get_audio_idx
from .text import split_sentences, split_sentences_from_input, Epub
from .sub import (
    extract_all_subtitles,
    cleanup_subfail,
    SUBTITLE_FORMATS,
    normalize_text,
)
from .utils import grab_files, get_tmp_path

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
VIDEO_FORMATS = [
    "3g2",
    "3gp",
    "avi",
    "flv",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "webm",
    "mpg",
]
TEXT_FORMATS = ["epub", "txt"]
WRITTEN_FORMATS = SUBTITLE_FORMATS + TEXT_FORMATS

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
        return duration
    except ffmpeg.Error as e:
        error_message = f"ffmpeg error: {e.stderr.decode('utf-8')}"
        raise RuntimeError(error_message)
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        raise RuntimeError(error_message)


class Writer:
    def __init__(self, output_format="srt"):
        self.written = False
        self.output_format = output_format

    def write_sub(self, segments, output_full_path):
        self.output_format = self.output_format
        with output_full_path.open("w", encoding="utf8") as o:
            if self.output_format == "srt":
                self.written = True
                return write_srt(segments, o)
            elif self.output_format == "vtt":
                self.written = True
                return write_vtt(segments, o)


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


@dataclass(eq=True, frozen=True)
class AudioSub(AudioStream):
    stream: ffmpeg.Stream
    audio_probe: dict
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
        audio_probe = get_audio_idx(info["streams"], lang, path)
        if whole or "chapters" not in info or len(info["chapters"]) == 0:
            stream = get_matching_audio_stream(info["streams"], lang)
            duration = get_video_duration(stream, path)
            return title, [
                cls(
                    stream=ffmpeg.input(path),
                    audio_probe=audio_probe,
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
                    path, ss=chapter["start_time"], to=chapter["end_time"]
                ),
                audio_probe=audio_probe,
                duration=float(chapter["end_time"]) - float(chapter["start_time"]),
                path=path,
                cn=chapter.get("tags", {}).get("title", ""),
                cid=chapter["id"],
                cache=cache,
            )
            for chapter in info["chapters"]
            if (float(chapter["end_time"]) - float(chapter["start_time"])) > 1
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
    alass: bool = False


def get_streams(audio, cache_inputs):
    streams = []
    for f in audio:
        basename_f = basename(f)
        title, audio_subs = AudioSub.from_file(f, cache_inputs)
        streams.append((basename_f, title, audio_subs))
    return streams


def get_chapters(text: List[str], lang, alass, nlp):
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
            split_sentences_from_input(
                [p.text() for p in epub.text()], txt_path, lang, nlp
            )

        elif file_ext in sub_exts:
            try:
                txt_path = normalize_text(file_path)
                if not alass:
                    split_sentences(txt_path, txt_path, lang, nlp)

            except Exception as e:
                logger.opt(exception=True).error(
                    f"❗Failed to normalize the sub '{file_name}'. We can't process them. Try to get subs from a different source and try again: {e}"
                )
                return []
            chapters.append((txt_path, [TextFile(path=txt_path, title=file_name)]))
        else:
            txt_path = get_tmp_path(
                Path(file_path).parent / f"{Path(file_path).stem}.txt"
            )
            if not alass:
                split_sentences(file_path, txt_path, lang, nlp)
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


def match_lang_ext_original(audios, texts, expected_lang_ext):
    grouped_files = defaultdict(list)
    audio_dict = {get_true_stem(Path(audio)): audio for audio in audios}

    for text in texts:
        subtitle_path = Path(text)
        true_stem = get_true_stem(subtitle_path)
        text_lang_ext = subtitle_path.stem.split(".")[-1]
        if true_stem in audio_dict and text_lang_ext == expected_lang_ext:
            grouped_files[audio_dict[true_stem]].append(subtitle_path)

    audios_filtered = []
    texts_filtered = []

    for audio, subs in grouped_files.items():
        if subs:
            audios_filtered.append(audio)
            texts_filtered.append(subs[0])
    return audios_filtered, texts_filtered


def match_files(audios, texts, folder, rerun, orig, alass=False):
    already_run = []
    if rerun:
        old = get_existing_rerun_files(folder, orig)
        text = grab_files(folder, ["*." + ext for ext in TEXT_FORMATS])
        already_run = os_sorted(list(set(text + old)))
        audios_filtered = audios
        texts_filtered = already_run
    elif orig:
        # We're only going to match up files that are already have the original lang ext to the audio file
        audios_filtered, texts_filtered = match_lang_ext_original(audios, texts, orig)
    else:
        # TODO: reconsider if any of this is worthwhile after switching to language code switches
        # already_run = get_existing_rerun_files(folder, orig)
        # already_run_text_paths = []
        # already_run_audio_paths = []
        # for ar in already_run:
        #     arPath = Path(ar)
        #     removed_second_stem = Path(arPath.stem).stem
        #     already_run_audio_paths.append(str(arPath.parent / removed_second_stem))
        #     already_run_text_paths.append(str(arPath.parent / arPath.stem))
        destemed_audio = [
            str(Path(audio).parent / Path(audio).stem) for audio in audios
        ]
        destemed_text = [str(Path(text).parent / Path(text).stem) for text in texts]
        audios_unique = list(set(destemed_audio))
        texts_unique = list(set(destemed_text))

        texts_filtered = [
            t for t in texts if str(Path(t).parent / Path(t).stem) in texts_unique
        ]
        audios_filtered = [
            a for a in audios if str(Path(a).parent / Path(a).stem) in audios_unique
        ]
    if len(audios_filtered) > 1 and len(texts_filtered) == 1 and len(already_run) == 0:
        logger.info(
            "🤔 Multiple audio files found, but only one text... If you use an epub, each chapter will be mapped to a single audio file"
        )
        return [audios_filtered], [[t] for t in texts_filtered]

    if len(audios_filtered) != len(texts_filtered):
        logger.warning(
            "🤔 The number of text files didn't match the number of audio files... Matching them based on sort order. You should probably double-check this."
        )

    return [[a] for a in audios_filtered], [[t] for t in texts_filtered]


def filter_sources_by_target_file(all_audios, all_texts, target_file: str | None):
    """
    Filters matched audio/text pairs based on a specific target media file.
    If target_file is None, it returns all pairs.
    """
    all_pairs = list(zip(all_audios, all_texts))

    if not target_file:
        return all_pairs # Return all pairs if no specific file is targeted

    logger.debug(f"Filtering sources to focus on target file: {Path(target_file).name}")
    target_file_path = Path(target_file)

    filtered_pairs = [
        (audio_list, text_list) for audio_list, text_list in all_pairs
        if Path(audio_list[0]) == target_file_path
    ]

    if not filtered_pairs:
        logger.warning(f"Target file '{target_file_path.name}' had no matching text")

    return filtered_pairs


def get_sources_from_dirs(input, cache_inputs, nlp):
    sources = []
    working_folders = get_working_folders(input.dirs)
    for folder in working_folders:
        audios = get_audio(folder)
        if input.subcommand == "sync":
            if input.alass:
                audios_to_process = audios
                target_file = getattr(input, 'file', None)
                if target_file:
                    target_path = Path(target_file)
                    audios_to_process = [a for a in audios if Path(a) == target_path]
                extract_all_subtitles(audios_to_process, input.lang_ext, input.lang_ext_original)
            texts = get_text(folder)
            a, t = match_files(
                audios, texts, folder, input.rerun, input.lang_ext_original
            )
            target_file = getattr(input, 'file', None)
            final_filtered_pairs = filter_sources_by_target_file(a, t, target_file)

            for matched_audio, matched_text in final_filtered_pairs:
                output_full_paths = get_output_full_paths(
                    matched_audio, folder, input.output_format, input.lang_ext
                )
                writer = Writer(input.output_format)

                streams = get_streams(matched_audio, cache_inputs)
                chapters = get_chapters(matched_text, input.lang, input.alass, nlp)
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
                    alass=input.alass,
                )
                sources.append(s)
        else:
            target_file = getattr(input, 'file', None)
            filtered_audios = audios
            if target_file:
                target_file_path = Path(target_file)
                filtered_audios = [
                    audio_path for audio_path in audios
                    if Path(audio_path) == target_file_path
                ]

            for matched_audio in filtered_audios:
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
                    alass=input.alass,
                )
                sources.append(s)
    return sources


def setup_sources(input, cache_inputs) -> List[sourceData]:
    nlp = None
    if hasattr(input, "nlp") and input.nlp:
        stanza.download(input.lang)
        nlp = stanza.Pipeline(lang=input.lang, processors="tokenize", use_gpu=False)
    if input.dirs:
        sources = get_sources_from_dirs(input, cache_inputs, nlp)
    else:
        if input.subcommand == "sync":
            if input.alass:
                extract_all_subtitles(
                    input.audio, input.lang_ext, input.lang_ext_original
                )
            output_dir = setup_output_dir(input.output_dir, input.audio[0])
            output_full_paths = get_output_full_paths(
                input.audio, output_dir, input.output_format, input.lang_ext
            )
            writer = Writer(input.output_format)
            chapters = get_chapters(input.text, input.lang, input.alass, nlp)
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
                    alass=input.alass,
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
                    alass=input.alass,
                )
            ]
    return sources


def rename_existing_file_to_old(p, orig):
    path_obj = Path(p)
    new_filename = path_obj.with_suffix(f".{orig}{path_obj.suffix}")
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
            old_file = get_rerun_file_path(op, input)
            if not source.overwrite and op.exists():
                logger.info(f"🤔 SubPlz file '{op.name}' already exists, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break
            if old_file.exists() and (str(old_file) == str(op)) and not source.rerun:
                logger.info(
                    f"🤔 {old_file.name} already exists but you don't want it overwritten, skipping. If you do, add --rerun"
                )
                invalid_sources.append(source)
                is_valid = False
                break
            if not source.audio:
                logger.error(f"❗ {op.name}'s audio is missing, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break
            if not source.text and input.subcommand == "sync":
                logger.error(f"❗ {source.audio}'s text is missing, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break
            if not source.chapters and input.subcommand == "sync":
                logger.error(f"❗ {source.text}'s couldn't be parsed, skipping.")
                invalid_sources.append(source)
                is_valid = False
                break
            if op.exists() and not old_file.exists() and input.subcommand == "sync":
                rename_existing_file_to_old(op, input.lang_ext_original)

        if is_valid:
            valid_sources.append(source)

    if input.subcommand == "sync":
        for source in valid_sources:
            logger.info(f"🎧 {pformat(source.audio)}' ➕ 📖 {pformat(source.text)}...")
        cleanup(invalid_sources)
    return valid_sources


def cleanup(sources: List[sourceData]):
    for source in sources:
        for file in source.text:
            for ext in WRITTEN_FORMATS:
                tmp_file = get_tmp_path(file).with_suffix(f".{ext}")
                tmp_file_path = Path(tmp_file)
                if tmp_file_path.exists():
                    tmp_file_path.unlink()


def get_existing_rerun_files(dir: str, orig) -> List[str]:
    old = grab_files(dir, [f"*.{orig}." + ext for ext in SUBTITLE_FORMATS])
    return old



def get_hearing_impaired_extensions() -> set:
    return {"cc", "hi", "sdh"}


def get_true_stem(file_path: Path) -> str:
    """
    Extract the base filename, removing subtitle-specific extensions.

    Handles patterns like:
    - filename.ja.cc.srt -> filename
    - filename.ja.srt -> filename
    - filename.en.srt -> filename
    - filename.mkv -> filename

    But preserves periods that are part of the actual title:
    - Title [Opus 2.0].mkv -> Title [Opus 2.0]
    """
    original_stem = file_path.stem

    # Don't modify stems that don't look like subtitle files
    # If there are no dots, or if it's a video file, return as-is
    if '.' not in original_stem or file_path.suffix.lower() in ['.mkv', '.mp4', '.avi', '.mov', '.m4v', '.webm', '.flv']:
        return original_stem

    # Work backwards from the end to remove subtitle extensions
    parts = original_stem.split('.')

    # Remove hearing impaired extensions (cc, hi, sdh) from the end
    hi_extensions = get_hearing_impaired_extensions()
    if len(parts) > 1 and parts[-1] in hi_extensions:
        parts = parts[:-1]

    # Remove language codes (1-3 character extensions) from the end
    if len(parts) > 1 and 1 <= len(parts[-1]) <= 3:
        # But only if it looks like a language code (letters only)
        if parts[-1].isalpha():
            parts = parts[:-1]

    return '.'.join(parts)


def get_rerun_file_path(output_path: Path, input) -> Path:
    orig_dot = ""
    if input.subcommand == "sync":
        if input.lang_ext_original:
            orig_dot = f".{input.lang_ext_original}"

    cache_file = (
        output_path.parent
        / f"{get_true_stem(output_path)}{orig_dot}{output_path.suffix}"
    )
    return cache_file


def rename_old_subs(source: sourceData, orig):
    subs = []
    for sub in source.text:
        if (
            Path(sub).suffix[1:] in SUBTITLE_FORMATS
            and f".{orig}" not in Path(Path(sub).stem).suffix
        ):
            subs.append(sub)
    remaining_subs = set(subs) - set([str(p) for p in source.output_full_paths])
    for i, sub in enumerate(remaining_subs):
        sub_path = Path(sub)

        if len(source.text) == len(source.audio):
            new_filename = Path(source.audio[i]).with_suffix(
                f".{orig}{sub_path.suffix}"
            )
        else:
            new_filename = sub_path.with_suffix(f".{orig}{sub_path.suffix}")
        sub_path.rename(new_filename)


def post_process(sources: List[sourceData], subcommand):
    if subcommand == "sync":
        cleanup(sources)
    complete_success = True
    sorted_sources = sorted(
        sources, key=lambda source: source.writer.written, reverse=True
    )
    for source in sorted_sources:
        if source.writer.written:
            output_paths = [str(o) for o in source.output_full_paths]
            logger.success(f"🙌 Successfully wrote '{', '.join(output_paths)}'")
        elif source.alass and not source.writer.written:
            complete_success = False
            logger.error(f"❗ Alass failed for '{source.audio[0]}'")
        else:
            complete_success = False
            logger.error(f"❗ Failed to sync '{source.text}'")

        cleanup_subfail(source.output_full_paths)
    alass_exists = (
        getattr(sources[0], "alass", None) if sources and len(sources) > 0 else None
    )
    if not sources:
        logger.warning(
            """😐 We didn't do anything. This may or may not be intentional. If this was unintentional, check if the destination file already exists"""
        )
    elif complete_success:
        logger.success("🎉 Everything went great!")
    else:
        if alass_exists:
            logger.error(
                """😭 At least one of the files failed to sync.
                Possible reasons:
                1. Alass failed to run or match subtitles
                2. Try using the `--lang-ext-incorrect ab` with the a sub with the same name as the media file, where `ab` is the language code in the sub extension
                3. Try using the `--lang-ext-original en` with the a sub with the same name as the media file, where `en` is the language code in the sub extension
                4. Try using the `--lang-ext ja` with the a sub with the same name as the media file, where `en` is the language code in the sub extension
                Something like this:

                subplz sync -d "/mnt/d/NeoOtaku Uprising The Anime" --alass --lang-ext "ja" --lang-ext-original "en" --lang-ext-incorrect "ab"

                /NeoOtaku Uprising The Anime/
                ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv (embedded subs can extract to Original)
                ├── NeoOtaku Uprising With Embedded Eng Subs EP00.en.srt (Original: Correct Timing)
                └── NeoOtaku Uprising With Embedded Eng Subs EP00.ab.srt (Japanese: Wrong Timings)
                This would generate for you
                NeoOtaku Uprising With Embedded Eng Subs EP00.ja.srt (Japanese: Correct Timings)
                """
            )
        else:
            logger.error(
                """😭 At least one of the files failed to sync.
                Possible reasons:
                1. The audio didn't match the text.
                2. The audio and text file matching might not have been ordered correctly.
                3. It could be cached - You could try running with `--overwrite-cache` if you've renamed another file to the exact same file path that you've run with the tool before.
                """
            )
