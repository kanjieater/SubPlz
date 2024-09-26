import os
from pathlib import Path
import ffmpeg
from subplz.utils import grab_files

SUBTITLE_FORMATS = ["ass", "srt", "vtt"]

def cleanup_subfail(output_paths):
    output_dirs = {path.parent for path in output_paths}
    subfail_files = []
    for output_dir in output_dirs:
        subfail_files.extend(grab_files(output_dir, ['*.subfail'], sort=False))

    for subfail_path in map(Path, subfail_files):
        successful_paths = {subfail_path.with_suffix(f'.{subtitle_format}') for subtitle_format in SUBTITLE_FORMATS}
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
            fail_file.write(f"Error processing subtitle for {source}:\n{error_message}\n")
        print(f"🚨 Error log written to {failed_path}")
    except Exception as e:
        print(f"❗ Failed to write error log to {failed_path}: {e}")



def get_subtitle_path(video_path, lang_ext):
    stem = Path(video_path).stem
    parent = Path(video_path).parent
    ext = f".{lang_ext}" if lang_ext else ""
    return parent / f"{stem}{ext}.srt"


def extract_subtitles(video_path: Path, output_subtitle_path: Path) -> None:
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(output_subtitle_path), map='0:s:0', c='srt', loglevel="quiet")
            .global_args("-hide_banner")
            .run(overwrite_output=True)
        )
        return output_subtitle_path
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to extract subtitles: {e.stderr.decode()}\nCommand: {str(e.cmd)}")


def extract_all_subtitles(files, lang_ext, lang_ext_original):
    for file in files:
        subtitle_path = get_subtitle_path(file, lang_ext_original)
        if subtitle_path.exists():
            continue
        try:
            print(f'⛏️ Extracting subtitles from {file} to {subtitle_path}')
            extract_subtitles(file, subtitle_path)
        except Exception as err:
            error_message = f"❗ Failed to extract subtitles; file not found: {subtitle_path}"
            print(err)
            target_subtitle_path = get_subtitle_path(file, lang_ext)
            write_subfail(file, target_subtitle_path, error_message)

