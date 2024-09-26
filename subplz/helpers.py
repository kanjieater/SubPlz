from typing import List
from pathlib import Path
from collections import defaultdict
import shutil
from subplz.files import SUPPORTED_AUDIO_FORMATS, get_true_stem, get_text, get_audio
from subplz.cli import CopyParams

def find(directories: List[str]) -> List[str]:
    audio_dirs = []

    for dir in directories:
        path = Path(dir)
        if path.is_dir():
            try:
                # Check the main directory for audio files
                if get_audio(path):  # Check if there are audio files in the directory
                    audio_dirs.append(str(path))

                for subdir in path.rglob("*"):
                    if subdir.is_dir() and get_audio(subdir):
                        audio_dirs.append(str(subdir))
            except OSError as e:
                print(f"Error accessing directory '{path}': {e}")

    print(audio_dirs)
    return audio_dirs


def get_rerun_file_path(output_path: Path, orig) -> Path:
    cache_file = output_path.parent / f"{get_true_stem(output_path)}.{orig}{output_path.suffix}"
    return cache_file

def rename(inputs):
    directories = inputs.dirs
    lang_ext = inputs.lang_ext
    lang_ext_original = inputs.lang_ext_original
    overwrite = inputs.overwrite

    for directory in directories:
        for text in get_text(directory):  # Get all subtitle files in the directory
            if f".{lang_ext_original}." in text:
                old_path = Path(text)  # Ensure we have a Path object
                true_stem = get_true_stem(old_path)

                # Create new name based on the stem, new language extension, and retain the original file extension
                new_name = old_path.parent / f"{true_stem}.{lang_ext}{old_path.suffix}"

                # Check if the new file exists
                if new_name.exists() and not overwrite:
                    print(f"Skipping renaming for {new_name} since it already exists.")
                    continue  # Use 'continue' instead of 'break' to keep processing other files

                # Attempt to rename the file
                try:
                    old_path.rename(new_name)
                    print(f"Renamed: {old_path} to {new_name}")
                except Exception as e:
                    print(f"Failed to rename {old_path} to {new_name}: {e}")

def copy(inputs: CopyParams):
 for directory in inputs.dirs:
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Skipping invalid directory: {directory}")
        continue
    audio_files = get_audio(dir_path)
    subtitle_files = get_text(dir_path)
    grouped_files = defaultdict(list)
    audio_dict = {get_true_stem(Path(audio)): audio for audio in audio_files}
    for subtitle in subtitle_files:
        subtitle_path = Path(subtitle)
        true_stem = get_true_stem(subtitle_path)
        if true_stem in audio_dict:
            grouped_files[audio_dict[true_stem]].append(subtitle)

    for audio, subs in grouped_files.items():
        copied = False
        for ext in inputs.lang_ext_priority:
            if copied:
                break

            for subtitle_file in subs[:]:
                old_path = Path(subtitle_file)
                if f".{ext}." in old_path.name:
                    true_stem = get_true_stem(old_path)
                    new_file = old_path.with_name(f"{true_stem}.{inputs.lang_ext}{old_path.suffix}")

                    if new_file.exists() and not inputs.overwrite:
                        print(f"Skipping copying {new_file} since it already exists")
                        subs.remove(subtitle_file)
                        copied = True
                        break

                    try:
                        shutil.copy(old_path, new_file)
                        print(f"Copied {old_path} to {new_file}")
                        subs.remove(subtitle_file)
                        copied = True
                        break
                    except Exception as e:
                        print(f"Failed to copy {old_path} to {new_file}: {e}")
                        copied = True
                        break