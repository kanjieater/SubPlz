from typing import List
from pathlib import Path
import shutil
from subplz.files import SUPPORTED_AUDIO_FORMATS, get_true_stem, get_text, get_audio
from subplz.cli import CopyParams

def find(directories: List[str]) -> List[str]:
    audio_dirs = []

    for dir in directories:
        path = Path(dir)
        if path.is_dir():
            try:
                if get_audio(path):
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

                if old_path.exists():
                    if new_name.exists() and not overwrite:
                        print(f"Skipping renaming for {new_name} since it already exists.")
                        break
                    old_path.rename(new_name)
                    print(f"Renamed: {old_path} to {new_name}")
                    break


def copy(inputs: CopyParams):
    for directory in inputs.dirs:
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Skipping invalid directory: {directory}")
            continue

        audio_files = get_audio(dir_path)
        subtitle_files = get_text(dir_path)

        for audio_file in audio_files:
            copied = False
            for ext in inputs.lang_ext_priority:
                if copied:
                    break
                for subtitle_file in subtitle_files:
                    old_path = Path(subtitle_file)
                    if f".{ext}." in old_path.name:
                        true_stem = get_true_stem(old_path)
                        new_file = old_path.with_name(f"{true_stem}.{inputs.lang_ext}{old_path.suffix}")

                        if new_file.exists() and not inputs.overwrite:
                            print(f"Skipping copying {new_file} since it already exists")
                            copied = True
                            break

                        try:
                            shutil.copy(old_path, new_file)
                            print(f"Copied {old_path} to {new_file}")
                            copied = True
                            break
                        except Exception as e:
                            print(f"Failed to copy {old_path} to {new_file}: {e}")