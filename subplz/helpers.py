import os
from pathlib import Path
import shutil
from subplz.files import SUPPORTED_AUDIO_FORMATS


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

def rename():
    """Rename subtitle files based on specified criteria."""
    inputs = get_inputs()
    rename_map = inputs.rename_map  # Assuming a rename_map is provided in the inputs

    for old_name, new_name in rename_map.items():
        old_path = Path(old_name)
        new_path = Path(new_name)
        if old_path.exists():
            old_path.rename(new_path)


def copy():
    """Copy subtitle files to a specified output directory."""
    inputs = get_inputs()
    output_dir = Path(inputs.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    subtitle_files = find()  # Use the find function to get the files
    for file in subtitle_files:
        shutil.copy(file, output_dir / file.name)