import re
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
import ffmpeg
from subplz.utils import grab_files
from dataclasses import dataclass
from subplz.utils import get_tmp_path, get_tqdm

tqdm, trange = get_tqdm()

SUBTITLE_FORMATS = ["ass", "srt", "vtt"]


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
                print(f"🧹 Removing '{subfail_path}' as we have a successful subtitle.")
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
    failed_path = target_path.with_suffix(".subfail")
    try:
        # Write the error message to the .subfail file
        with failed_path.open("w") as fail_file:
            fail_file.write(
                f"Error processing subtitle for {source}:\n{error_message}\n"
            )
        print(f"🚨 Error log written to {failed_path}")
    except Exception as e:
        print(f"❗ Failed to write error log to {failed_path}: {e}")


def get_subtitle_path(video_path, lang_ext):
    stem = Path(video_path).stem
    parent = Path(video_path).parent
    ext = f".{lang_ext}" if lang_ext else ""
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


def convert_sub_format(full_original_path, full_sub_path):
    stream = ffmpeg.input(full_original_path)
    stream = ffmpeg.output(stream, full_sub_path, loglevel="error").global_args(
        "-hide_banner"
    )
    ffmpeg.run(stream, overwrite_output=True)

def convert_between_sub_format(full_original_path, full_sub_path, format='srt'):
    stream = ffmpeg.input(full_original_path)
    # Pass the format as a keyword argument
    stream = ffmpeg.output(stream, full_sub_path, format=format, loglevel='error').global_args("-hide_banner")
    ffmpeg.run(stream, overwrite_output=True)


def normalize_text(file_path):
    file_path = Path(file_path)
    filename = file_path.stem
    srt_path = get_tmp_path(file_path.parent / f"{filename}.srt")
    txt_path = get_tmp_path(file_path.parent / f"{filename}.txt")
    convert_sub_format(str(file_path), str(srt_path))
    txt_path = remove_timing_and_metadata(srt_path, txt_path)
    srt_path.unlink()
    return str(txt_path)


def normalize_sub(file_path):
    file_path = Path(file_path)
    filename = file_path.stem
    vtt_path = get_tmp_path(file_path.parent / f"{filename}.ass")
    convert_between_sub_format(str(file_path), str(vtt_path), format="ass")
    convert_between_sub_format(str(vtt_path), str(file_path))
    vtt_path.unlink()
    return str(file_path)


def sanitize_subtitle(subtitle_path: Path) -> None:
    try:
        normalize_sub(subtitle_path)
        print(f"🧼 Sanitized subtitles at {subtitle_path}")

    except Exception as e:
        print(f"❗ Failed to sanitize subtitles: {e}")


def ffmpeg_extract(video_path: Path, output_subtitle_path: Path) -> None:
    # Need stdin flag for running from bash here
    # https://stackoverflow.com/questions/25811022/incorporating-ffmpeg-in-a-bash-script
    try:
        (
            ffmpeg.input(str(video_path))
            .output(str(output_subtitle_path), map="0:s:0", c="srt", loglevel='error')
            .global_args("-hide_banner", "-nostdin")
            .run(overwrite_output=True)
        )
        return output_subtitle_path
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Failed to extract subtitles: {e.stderr.decode()}\nCommand: {str(e.cmd)}"
        )


def extract_subtitle(file, lang_ext, lang_ext_original):
    subtitle_path = get_subtitle_path(file, lang_ext_original)
    if subtitle_path.exists():
        return

    try:
        print(f"⛏️ Extracting subtitles from {file} to {subtitle_path}")
        ffmpeg_extract(file, subtitle_path)
    except Exception as err:
        error_message = (
            f"❗ Failed to extract subtitles; file not found: {subtitle_path}"
        )
        print(err)
        target_subtitle_path = get_subtitle_path(file, lang_ext)
        write_subfail(file, target_subtitle_path, error_message)


def extract_all_subtitles(files, lang_ext, lang_ext_original):
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(partial(extract_subtitle, lang_ext=lang_ext, lang_ext_original=lang_ext_original), files),
                   total=len(files),
                   desc="Extracting Subtitles in Parallel (may take a while)"))
