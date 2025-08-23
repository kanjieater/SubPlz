import re
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import ffmpeg
from .logger import logger
from .utils import grab_files, get_tmp_path, get_tqdm, get_lang_code
import concurrent.futures

tqdm, trange = get_tqdm()

SUBTITLE_FORMATS = ["ass", "srt", "vtt", "ssa", "idx"]


def score_subtitle_stream(stream_info: dict, target_iso_lang: str | None) -> int:
    """Assigns a preference score to a subtitle stream."""
    score = 0
    tags = stream_info.get("tags", {})
    title = tags.get("title", "").lower()
    stream_lang = tags.get("language", "").lower()
    disposition = stream_info.get("disposition", {})
    codec_name = stream_info.get("codec_name", "").lower()

    image_based_codecs = [
        "hdmv_pgs_subtitle",  # For Blu-ray (PGS)
        "dvd_subtitle",  # For DVD (VOBSUB)
        "dvb_subtitle",  # For DVB broadcasts
    ]
    if codec_name in image_based_codecs:
        score -= 500

    # 1. Language Match (highest importance)
    if target_iso_lang and stream_lang == target_iso_lang:
        score += 100

    # 2. Avoid undesirable tracks (strong negative scores)
    commentary_keywords = ["commentary", "comment", "comms"]
    signs_songs_keywords = ["signs", "songs", "s&s", "sign", "song"]
    forced_keywords = ["forced"]

    if any(kw in title for kw in commentary_keywords) or disposition.get("comment", 0):
        score -= 200
    if any(kw in title for kw in signs_songs_keywords):
        score -= 150
    if any(kw in title for kw in forced_keywords) or disposition.get("forced", 0):
        score -= 100

    # 3. Prefer desirable tracks (positive scores)
    full_keywords = ["full", "dialogue", "dialog"]
    sdh_keywords = ["sdh", "cc", "hearing impaired"]

    if any(kw in title for kw in full_keywords):
        score += 20
    if any(kw in title for kw in sdh_keywords):
        score += 15

    # 4. Prefer default track (good tie-breaker)
    if disposition.get("default", 0):
        score += 10

    # 5. Format preference (minor tie-breaker)
    codec_name = stream_info.get("codec_name", "").lower()
    if codec_name == "srt":
        score += 5
    elif codec_name == "ass":
        score += 2

    return score


def get_subtitle_idx(
    all_streams: list, path: str, target_lang_code: str = None
) -> dict | None:
    """
    Finds the best matching subtitle stream.
    If target_lang_code is provided, it will be strict and only return a match for that language.
    If target_lang_code is None, it will return the highest-scoring stream regardless of language.
    """
    subtitle_streams = [s for s in all_streams if s.get("codec_type") == "subtitle"]
    if not subtitle_streams:
        return None

    standardized_target_lang = None
    if target_lang_code:
        standardized_target_lang = get_lang_code(target_lang_code)
        if not standardized_target_lang:
            logger.warning(
                f"🦈 Could not standardize input language code '{target_lang_code}'."
            )

    scored_streams = []
    for stream in subtitle_streams:
        score = score_subtitle_stream(stream, standardized_target_lang)
        scored_streams.append({"score": score, "stream_info": stream})

    positive_streams = [s for s in scored_streams if s["score"] > 0]
    if not positive_streams:
        logger.warning(f"🦈 No subtitle stream scored positively for {path}")
        return None
    positive_streams.sort(key=lambda x: x["score"], reverse=True)

    best_match = positive_streams[0]
    best_match_lang = (
        best_match["stream_info"].get("tags", {}).get("language", "").lower()
    )

    if standardized_target_lang and best_match_lang != standardized_target_lang:
        logger.warning(
            f"🦈 No subtitle stream with the required language '{target_lang_code}' was found in {path}"
        )
        return None

    logger.info(
        f"字 Selected subtitle stream (Index: {best_match['stream_info'].get('index')}, Score: {best_match['score']}, Codec: {best_match['stream_info'].get('codec_name', 'N/A')}, Lang: {best_match_lang or 'N/A'}, Title: {best_match['stream_info'].get('tags',{}).get('title','N/A')}) for file: {path}"
    )
    return best_match["stream_info"]


def sexagesimal(secs, use_comma=False):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    r = f"{hh:02}:{mm:02}:{ss:06.3f}"
    if use_comma:
        r = r.replace(".", ",")
    return r


@dataclass(eq=True)
class Segment:
    text: str
    start: float
    end: float

    def __repr__(self):
        return f"Segment(text='{self.text}', start={sexagesimal(self.start)}, end={sexagesimal(self.end)})"

    def vtt(self, use_comma=False):
        return f"{sexagesimal(self.start, use_comma)} --> {sexagesimal(self.end, use_comma)}\n{self.text}"


def write_srt(segments, o):
    srt_content = "\n\n".join(
        f"{i + 1}\n{s.vtt(use_comma=True)}" for i, s in enumerate(segments)
    )
    o.write(srt_content)


def write_vtt(segments, o):
    vtt_content = "WEBVTT\n\n" + "\n\n".join(s.vtt() for s in segments)
    o.write(vtt_content)


def cleanup_subfail(output_paths):
    output_dirs = {path.parent for path in output_paths}
    subfail_files = []
    for output_dir in output_dirs:
        subfail_files.extend(grab_files(output_dir, ["*.subfail"], sort=False))

    for subfail_path in map(Path, subfail_files):
        successful_paths = {
            subfail_path.with_suffix(f".{subtitle_format}")
            for subtitle_format in SUBTITLE_FORMATS
        }
        for successful_path in successful_paths:
            if successful_path.exists():
                logger.info(
                    f"🧹 Removing '{subfail_path}' as we have a successful subtitle."
                )
                os.remove(subfail_path)


def write_subfail(source, target_path, error_message):
    """
    Writes an error message to a file indicating subtitle processing failure.
    The file will have the same base name as the target subtitle path but with a `.subfail` extension.

    :param source: The source stream object for logging details.
    :param target_path: The path where the target subtitle would be saved.
    :param error_message: The error message to log in the `.subfail` file.
    """
    # Define the path for the failure file
    failed_path = Path(target_path).with_suffix(".subfail")
    try:
        # Use UTF-8 encoding for safety
        with failed_path.open("w", encoding="utf-8") as fail_file:
            fail_file.write(
                f"Error processing subtitle for {source}:\n{error_message}\n"
            )
        logger.error(f"🚨 Error log written to {failed_path}")
    except Exception as e:
        logger.critical(f"❗ Failed to write error log to {failed_path}: {e}")


def get_subtitle_path(video_path, lang_ext):
    stem = Path(video_path).stem
    parent = Path(video_path).parent
    ext = f".{lang_ext}" if lang_ext else ""
    for format in SUBTITLE_FORMATS:
        if (parent / f"{stem}{ext}.{format}").exists():
            return parent / f"{stem}{ext}.{format}"
    return parent / f"{stem}{ext}.srt"


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


def check_empty_subs(subtitle_path):
    """
    Checks if a subtitle file is empty or invalid. Handles str or Path objects gracefully.
    """
    path_obj = Path(subtitle_path)

    try:
        # Use the Path object for all operations.
        if not path_obj.exists() or path_obj.stat().st_size < 10:
            if path_obj.exists():
                path_obj.unlink()
            raise ValueError("❗Subtitle file is empty or invalid.")
    except Exception as e:
        # Now, using .name is safe because path_obj is guaranteed to be a Path.
        raise RuntimeError(
            f"❗Validation failed for subtitle file '{path_obj.name}': {e}"
        ) from e


def convert_between_sub_format(full_original_path, full_sub_path, format="srt"):
    original_path_str = str(full_original_path)
    sub_path_str = str(full_sub_path)

    # Use the string versions in the ffmpeg calls.
    stream = ffmpeg.input(original_path_str)
    stream = ffmpeg.output(
        stream, sub_path_str, format=format, loglevel="error"
    ).global_args("-hide_banner")
    ffmpeg.run(stream, overwrite_output=True)


def normalize_text(file_path):
    file_path = Path(file_path)
    filename = file_path.stem
    srt_path = get_tmp_path(file_path.parent / f"{filename}.srt")
    txt_path = get_tmp_path(file_path.parent / f"{filename}.txt")
    convert_between_sub_format(file_path, srt_path)
    txt_path = remove_timing_and_metadata(srt_path, txt_path)
    srt_path.unlink()
    return str(txt_path)


def normalize_sub(file_path):
    file_path = Path(file_path)
    filename = file_path.stem
    vtt_path = get_tmp_path(file_path.parent / f"{filename}.ass")
    convert_between_sub_format(file_path, vtt_path, format="ass")
    convert_between_sub_format(vtt_path, file_path)
    vtt_path.unlink()
    return str(file_path)


def sanitize_subtitle(subtitle_path: Path) -> None:
    """
    Checks if a subtitle file is valid and normalizes its format.
    Raises an error if the file is empty or invalid.
    """

    try:
        normalize_sub(subtitle_path)
        logger.debug(f"🧼 Sanitized subtitles at {subtitle_path}")
    except Exception as e:
        # If normalization fails, it's also a critical error
        raise RuntimeError(
            f"❗Failed to normalize subtitles for '{subtitle_path.name}': {e}"
        ) from e


def ffmpeg_extract(
    video_path: Path, output_subtitle_path: Path, stream_index: int
) -> Path:
    """
    Extracts a subtitle stream safely using the tempfile module and an atomic rename.
    """
    final_path = output_subtitle_path

    temp_path = get_tmp_path(final_path)

    try:
        stream_specifier = f"0:s:{stream_index}"
        print(f"Extracting to temporary file: {temp_path}")
        (
            ffmpeg.input(
                str(video_path),
                # Tell ffmpeg to analyze only the first 10MB of the file to find streams.
                # This avoids scanning the entire file over the network.
                probesize="10M",
                # Tell ffmpeg to analyze up to 10 seconds of video duration.
                # Value is in microseconds.
                analyzeduration="10000000",
            )
            .output(
                str(temp_path),
                map=stream_specifier,
                c="srt",
                loglevel="error",
            )
            .global_args("-hide_banner", "-nostdin")
            .run(overwrite_output=True)
        )
        # check_empty_subs(temp_path)
        # Atomic rename from temp to final location
        temp_path.rename(final_path)
        return final_path
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"❗Failed to extract subtitle stream index {stream_index} from {video_path}. FFmpeg error: {e.stderr.decode()}"
            "You may need to provide an external subtitle file."
            "If the error above Stream map '0:s:0' matches no streams, then no embedded subtitles found in the file"
        )
    finally:
        temp_path.unlink(missing_ok=True)


def extract_subtitle(
    file,
    lang_ext,
    lang_ext_original,
    overwrite=False,
    existence_check_lang=None,
    strict=False,
):
    lang_to_check = (
        existence_check_lang if existence_check_lang is not None else lang_ext_original
    )
    path_to_check = get_subtitle_path(file, lang_to_check)
    output_subtitle_path = get_subtitle_path(file, lang_to_check)

    if path_to_check.exists() and not overwrite:
        logger.info(
            f"☑️ Subtitle '{path_to_check.name}' already exists, skipping extraction."
        )
        return None

    try:
        logger.info(
            f"🔎 Analyzing streams in {file} to find best '{lang_ext_original}' subtitle..."
        )
        all_streams = ffmpeg.probe(str(file))["streams"]
        strict_lang = None
        if strict:
            strict_lang = lang_ext_original
        best_sub_stream = get_subtitle_idx(all_streams, str(file), strict_lang)

        if not best_sub_stream:
            logger.info(
                f"🤷 No suitable subtitle streams found '{lang_ext_original}' in {file}"
            )
            return None

        stream_index_in_file = best_sub_stream["index"]
        subtitle_only_streams = [
            s for s in all_streams if s.get("codec_type") == "subtitle"
        ]
        relative_stream_index = 0
        for i, s in enumerate(subtitle_only_streams):
            if s["index"] == stream_index_in_file:
                relative_stream_index = i
                break

        logger.info(
            f"⛏️ Extracting best subtitle stream (index: {relative_stream_index}) from {file} to {output_subtitle_path}"
        )
        ffmpeg_extract(file, output_subtitle_path, relative_stream_index)

        return output_subtitle_path

    except Exception as err:
        error_message = f"❗ Failed to extract subtitles from {file}: {err}"
        logger.error(error_message)
        write_subfail(file, output_subtitle_path, error_message)
        return None


def extract_all_subtitles(
    files,
    lang_ext,
    lang_ext_original,
    overwrite=False,
    existence_check_lang=None,
    strict=False,
):
    # Using a process count of 1 is safer for network drives with processes
    # Using a thread pool is safer than a process pool when the parent process has an active CUDA context.
    # Setting max_workers=1 makes it run serially, avoiding parallelism for now.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        func = partial(
            extract_subtitle,
            lang_ext=lang_ext,
            lang_ext_original=lang_ext_original,
            overwrite=overwrite,
            existence_check_lang=existence_check_lang,
            strict=strict,
        )
        results = list(
            tqdm(
                executor.map(func, files),
                total=len(files),
                desc=f"Checking if '{lang_ext_original}' exists or if we should extract to make '{lang_ext}'",
            )
        )
    extracted_paths = [path for path in results if path is not None]
    return extracted_paths
