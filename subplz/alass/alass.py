import subprocess
from pathlib import Path
from subplz.utils import get_tqdm
from subplz.sub import (
    get_subtitle_path,
    write_subfail,
    sanitize_subtitle,
    convert_between_sub_format,
)

tqdm, trange = get_tqdm()

# Define paths
alass_dir = Path(__file__).parent
alass_path = alass_dir / "alass-linux64"


# FIX 1: Add the custom exception class definition
class SubtitleProcessingError(Exception):
    pass


def perform_srt_conversion(subtitle_path):
    """
    Takes a subtitle path, converts it to a temporary SRT file, and returns the new path.
    This function assumes a conversion is necessary.
    """
    try:
        print(f"🔄 Converting {subtitle_path.name} to SRT format...")
        temp_path = subtitle_path.with_suffix(".tmp.srt")
        convert_between_sub_format(str(subtitle_path), str(temp_path), format="srt")
        # It successfully returns just one value: the path to the new temp file.
        return temp_path
    except Exception as err:
        raise SubtitleProcessingError(
            f"Subtitle conversion to SRT failed for {subtitle_path.name}: {err}"
        ) from err


def validate_subtitle_paths(og_subtitle_path, incorrect_subtitle_path):
    """
    Validates the existence and uniqueness of the necessary subtitle paths.
    """
    if not og_subtitle_path or not og_subtitle_path.exists():
        return f"❗ Skipping sync: Original subtitle not found: {og_subtitle_path}"

    if not incorrect_subtitle_path or not incorrect_subtitle_path.exists():
        return f"❗ Subtitle with incorrect timing not found: {incorrect_subtitle_path}"

    if str(og_subtitle_path) == str(incorrect_subtitle_path):
        return "❗ Skipping sync: Original and incorrect subtitles are the same file."

    return None  # Indicates success


def run_alass_alignment(
    og_subtitle_path, incorrect_subtitle_path, target_subtitle_path
):
    """
    Executes the alass command-line tool to align subtitles.
    """
    print(
        f"🤝 Aligning {incorrect_subtitle_path.name} based on {og_subtitle_path.name}"
    )
    cmd = [
        str(alass_path),
        str(og_subtitle_path),
        str(incorrect_subtitle_path),
        str(target_subtitle_path),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        error_message = (
            f"❗ Alass command failed with code {result.returncode}\n\n"
            f"STDOUT: {result.stdout}\n\n"
            f"STDERR: {result.stderr}"
        )
        return False, error_message
    else:
        return True, result.stdout


def sync_alass(source, input_sources, be):
    """
    Orchestrates the subtitle synchronization process with inlined logic.
    """
    with tqdm(source.streams) as bar:
        for batches in bar:
            video_path = batches[2][0].path
            temp_files_to_clean = []
            target_subtitle_path = None

            try:
                # --- Path discovery ---
                og_subtitle_path = get_subtitle_path(
                    video_path, input_sources.lang_ext_original
                )
                incorrect_subtitle_path = get_subtitle_path(
                    video_path, input_sources.lang_ext_incorrect
                )
                target_subtitle_path = get_subtitle_path(
                    video_path, input_sources.lang_ext
                )

                # --- Step 1: Validation ---
                validation_error = validate_subtitle_paths(
                    og_subtitle_path, incorrect_subtitle_path
                )
                if validation_error:
                    print(validation_error)
                    write_subfail(source, target_subtitle_path, validation_error)
                    continue

                # --- Step 2a: Sanitization ---
                try:
                    sanitize_subtitle(og_subtitle_path)
                    sanitize_subtitle(incorrect_subtitle_path)
                except Exception as err:
                    error_message = f"❗ Subtitle sanitization failed: {err}"
                    print(error_message)
                    write_subfail(source, target_subtitle_path, error_message)
                    continue

                # --- Step 2b: Conversion ---
                try:
                    og_sub = og_subtitle_path
                    if og_subtitle_path.suffix != ".srt":
                        temp_og_path = perform_srt_conversion(og_subtitle_path)
                        og_sub = temp_og_path
                        temp_files_to_clean.append(temp_og_path)

                    incorrect_sub = incorrect_subtitle_path
                    if incorrect_subtitle_path.suffix != ".srt":
                        temp_incorrect_path = perform_srt_conversion(
                            incorrect_subtitle_path
                        )
                        incorrect_sub = temp_incorrect_path
                        temp_files_to_clean.append(temp_incorrect_path)
                except Exception as err:
                    error_message = f"❗ Subtitle sanitization failed: {err}"
                    print(error_message)
                    write_subfail(source, target_subtitle_path, error_message)
                    continue
                # --- Step 3: Alignment ---
                success, message = run_alass_alignment(
                    og_sub, incorrect_sub, target_subtitle_path
                )

                if success:
                    source.writer.written = True
                else:
                    print(message)
                    write_subfail(source, target_subtitle_path, message)

            finally:
                # --- Cleanup ---
                for temp_file in temp_files_to_clean:
                    if temp_file and temp_file.exists():
                        temp_file.unlink()
