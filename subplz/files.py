import os
from pathlib import Path
from os import path
from ats.main import TextFile, Epub, AudioStream

def get_streams(audio):
    print("Loading streams...")
    streams = [(os.path.basename(f), *AudioStream.from_file(f)) for f in audio]
    return streams

def get_chapters(text):
    print("Finding chunks...")
    chapters = [(os.path.basename(i), Epub.from_file(i)) if i.split(".")[-1] == 'epub' else (os.path.basename(i), [TextFile(path=i, title=os.path.basename(i))]) for i in text]
    return chapters


def get_content_name(working_folder):
    folder_name = path.dirname(working_folder)
    content_name = path.basename(folder_name)
    return content_name

def get_working_folders(dirs):
    working_folders = []
    for dir in dirs:
        full_folder = path.join(dir, "")
        content_name = get_content_name(dir)
        # split_folder = path.join(full_folder, f"{content_name}_splitted")

        # if path.exists(split_folder) and path.isdir(split_folder):
        #     working_folder = split_folder
        #     print(
        #         f"Warning: Using split files causes a fixed delay for every split file. This is a known bug. Use the single file method instead"
        #     )
        # else:
        working_folder = full_folder
        working_folders.append(working_folder)
    return working_folders