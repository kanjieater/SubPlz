import os
from pathlib import Path
import shutil
from subplz.files import SUPPORTED_AUDIO_FORMATS, get_true_stem, get_text


def find(directories):
    all_directories = []

    for dir in directories:
        path = Path(dir)
        if path.is_dir():
            try:
                # Recursively find all directories
                for subdir in path.rglob("*"):
                    if subdir.is_dir():
                        all_directories.append(str(subdir))
            except OSError as e:
                print(f"Error accessing directory '{path}': {e}")

    print(all_directories)
    return all_directories


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




def copy():
    """Copy subtitle files to a specified output directory."""
    inputs = get_inputs()
    output_dir = Path(inputs.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    subtitle_files = find()  # Use the find function to get the files
    for file in subtitle_files:
        shutil.copy(file, output_dir / file.name)