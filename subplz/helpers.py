#extract
from subplz.sub import extract_all_subtitles
from subplz.files import get_audio

from typing import List
from pathlib import Path
from collections import defaultdict
import shutil
from subplz.files import (
    SUPPORTED_AUDIO_FORMATS,
    get_true_stem,
    get_text,
    get_audio,
    match_files,
    get_audio,
)
from subplz.cli import CopyParams, ExtractParams
from subplz.text import detect_language



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
    cache_file = (
        output_path.parent / f"{get_true_stem(output_path)}.{orig}{output_path.suffix}"
    )
    return cache_file


def rename(inputs):
    directories = inputs.dirs
    lang_ext = inputs.lang_ext
    lang_ext_original = inputs.lang_ext_original
    overwrite = inputs.overwrite
    if not lang_ext:
        print(
            "❗ Failed to rename. You must include a language extension --lang-ext to add to the output file name."
        )
        return

    rename_texts = []
    if lang_ext_original is None:
        for directory in directories:
            texts = get_text(directory)
            audios = get_audio(directory)
            matched_audios, matched_texts = match_files(
                audios, texts, directory, False, None
            )
            for audio, text in zip(matched_audios, matched_texts):
                audio_path = Path(audio[0])
                text_path = Path(text[0])
                true_stem = get_true_stem(audio_path)
                new_name = (
                    audio_path.parent / f"{true_stem}.{lang_ext}{text_path.suffix}"
                )
                rename_texts.append({str(text_path): new_name})

    else:
        for directory in directories:
            for text in get_text(directory):
                if f".{lang_ext_original}." in text:
                    text_path = Path(text)
                    true_stem = get_true_stem(text_path)
                    new_name = (
                        text_path.parent / f"{true_stem}.{lang_ext}{text_path.suffix}"
                    )
                    rename_texts.append({str(text_path): new_name})

    for rename_text in rename_texts:
        text_path, new_name = list(rename_text.items())[0]
        old_path = Path(text_path)
        if new_name.exists() and not overwrite:
            print(f"😐 Skipping renaming for {new_name} since it already exists.")
            continue
        try:
            print(f"""
                  {old_path}
                  ⚠️ is being renamed to ⚠️
                  {new_name}""")
            if not inputs.dry_run:
                old_path.rename(new_name)
        except Exception as e:
            print(f"❗ Failed to rename {old_path} to {new_name}: {e}")


def copy(inputs: CopyParams):
    for directory in inputs.dirs:
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"❗Skipping invalid directory: {directory}")
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

                for subtitle_file in subs:
                    old_path = Path(subtitle_file)
                    if f".{ext}." in old_path.name:
                        true_stem = get_true_stem(old_path)
                        new_file = old_path.with_name(
                            f"{true_stem}.{inputs.lang_ext}{old_path.suffix}"
                        )

                        if new_file.exists() and not inputs.overwrite:
                            print(
                                f"Skipping copying {new_file} since it already exists"
                            )
                            copied = True
                            break

                        try:
                            shutil.copy(old_path, new_file)
                            print(f"Copied {old_path} to {new_file}")
                            copied = True
                            break
                        except Exception as e:
                            print(f"Failed to copy {old_path} to {new_file}: {e}")
                            copied = True
                            break



def extract(inputs: ExtractParams):
    """
    Extracts embedded subtitles from media files by wrapping the core extract_all_subtitles function.
    """
    # --- 1. Validate Inputs ---
    if not inputs.lang_ext:
        print("❗ --lang-ext is required to specify the output subtitle language extension.")
        return
    if not inputs.lang_ext_original:
        print("❗ --lang-ext-original is required to specify the language to search for and extract.")
        return

    # --- 2. Discover Files ---
    media_files = []
    for directory in inputs.dirs:
        dir_path = Path(directory)
        if dir_path.is_dir():
            # get_audio finds all supported media containers (video and audio)
            media_files.extend(get_audio(dir_path))
        else:
            print(f"❗ Skipping invalid directory: {dir_path}")

    if not media_files:
        print("🤷 No media files found to process.")
        return

    # --- 3. Call the Reused Function ---
    # This now calls the core logic from sub.py with all the necessary parameters.
    extracted_subs = extract_all_subtitles(
        files=media_files,
        lang_ext=inputs.lang_ext,
        lang_ext_original=inputs.lang_ext_original,
        overwrite=inputs.overwrite,
        existence_check_lang=inputs.lang_ext
    )
    if not extracted_subs:
        print("🤔 No new subtitles were extracted")
        return
    # --- 4. Verify Language if Requested ---
    if not inputs.verify:
        return

    print("🕵️ Verifying language of newly extracted files...")
    for sub_path in extracted_subs:
        detected_lang = detect_language(sub_path)
        if detected_lang != inputs.lang_ext_original:
            print(f"❌ Language mismatch for '{sub_path.name}'! Expected '{inputs.lang_ext_original}', detected '{detected_lang}'.")
            sub_path.unlink()
        else:
            print(f"✅ Language verified for '{sub_path.name}'.")
