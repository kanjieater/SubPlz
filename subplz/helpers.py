from typing import List
from pathlib import Path
from collections import defaultdict
import shutil
from .logger import logger
from .sub import extract_all_subtitles
from .files import get_true_stem, get_text, get_audio, match_files
from .cli import CopyParams, ExtractParams, RenameParams
from .text import detect_language


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


def are_files_identical(file1: Path, file2: Path) -> bool:
    """
    Compares the content of two files to see if they are identical.
    Returns True if they match, False otherwise.
    """
    try:
        if file1.stat().st_size != file2.stat().st_size:
            return False  # Quick check: if sizes differ, they can't be identical
        if file1.read_text(encoding="utf-8") == file2.read_text(encoding="utf-8"):
            return True
        return False
    except Exception as e:
        logger.warning(f"⚠️  Could not compare files {file1.name} and {file2.name}: {e}")
        return False


def rename(inputs: RenameParams):
    directories = inputs.dirs
    lang_ext = inputs.lang_ext
    lang_ext_original = inputs.lang_ext_original
    overwrite = inputs.overwrite
    if not lang_ext:
        logger.error(
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
    target_texts = []
    target_file = getattr(inputs, "file", None)

    if target_file and Path(target_file).is_file():
        target_stem = get_true_stem(Path(target_file))
        logger.debug(
            f"Filtering operations to only process files matching stem: '{target_stem}'"
        )

        # Filter the list to only include operations where the source file's stem matches the target's stem
        target_texts = [
            t
            for t in rename_texts
            if get_true_stem(Path(list(t.keys())[0])) == target_stem
        ]

        if not target_texts:
            logger.warning(
                f"🤷 No subtitle files with the stem '{target_stem}' were found to rename."
            )
            return
    else:
        target_texts = rename_texts
    for rename_text in target_texts:
        text_path, new_name = list(rename_text.items())[0]
        old_path = Path(text_path)
        if new_name.exists() and not overwrite:
            logger.info(
                f"😐 Skipping rename for '{new_name.name}': Target file already exists and overwrite is false."
            )
            continue

        if inputs.unique:
            dir_path = old_path.parent
            true_stem = get_true_stem(old_path)
            if overwrite and lang_ext_original:
                original_base_file_path = (
                    dir_path / f"{true_stem}.{lang_ext_original}{old_path.suffix}"
                )
                if (
                    new_name.exists()
                    and original_base_file_path.exists()
                    and old_path != original_base_file_path
                    and are_files_identical(old_path, new_name)
                ):
                    logger.warning(
                        f"🚮 Deleting redundant source file '{old_path.name}'. "
                        f"It is identical to the target '{new_name.name}' and '{original_base_file_path.name}' exists already."
                    )
                    if not inputs.dry_run:
                        old_path.unlink()
                    continue

            is_unique = True
            all_subs_in_dir = get_text(dir_path)
            other_subs = [
                Path(p)
                for p in all_subs_in_dir
                if get_true_stem(Path(p)) == true_stem and Path(p) != old_path
            ]
            for other_sub in other_subs:
                if are_files_identical(old_path, other_sub):
                    print(
                        f"😐 Skipping rename for '{old_path}': Content is identical to '{other_sub}'."
                    )
                    is_unique = False
                    break
        if not is_unique:
            continue

        try:
            logger.info(
                f"""Renaming file:
                  Source: {old_path}
                  Target: {new_name}"""
            )
            if not inputs.dry_run:
                old_path.rename(new_name)
        except Exception as e:
            logger.error(f"❗ Failed to rename {old_path} to {new_name}: {e}")


def copy(inputs: CopyParams):
    for directory in inputs.dirs:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning(f"❗ Skipping invalid directory: {directory}")
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

        groups_to_process = grouped_files
        target_file = getattr(inputs, "file", None)
        if target_file and Path(target_file).is_file():
            target_stem = get_true_stem(Path(target_file))
            logger.debug(
                f"Filtering copy operations to only process files matching stem: '{target_stem}'"
            )
            groups_to_process = {
                audio: subs
                for audio, subs in grouped_files.items()
                if get_true_stem(Path(audio)) == target_stem
            }

            if not groups_to_process:
                logger.warning(
                    f"🤷 No media files with the stem '{target_stem}' were found in '{directory}' to process for copying."
                )
                continue  # Move to the next directory

        for audio, subs in groups_to_process.items():
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
                            logger.info(
                                f"😐 Skipping copy for '{new_file}' since it already exists."
                            )
                            copied = True
                            break

                        try:
                            shutil.copy(old_path, new_file)
                            logger.success(f"✅ Copied {old_path} to {new_file}")
                            copied = True
                            break
                        except Exception as e:
                            logger.error(
                                f"❗ Failed to copy {old_path} to {new_file}: {e}"
                            )
                            copied = True
                            break


def extract(inputs: ExtractParams):
    """
    Extracts embedded subtitles from media files by wrapping the core extract function.
    Respects the --file argument for targeted extraction.
    """
    if not inputs.lang_ext:
        logger.error(
            "❗ --lang-ext is required to specify the output subtitle language extension."
        )
        return
    if not inputs.lang_ext_original:
        logger.error(
            "❗ --lang-ext-original is required to specify the language to search for and extract."
        )
        return

    media_files = []
    target_file = getattr(inputs, "file", None)
    if target_file and Path(target_file).is_file():
        logger.info(f"Targeting specific file for extraction: {Path(target_file).name}")
        media_files.append(target_file)
    else:
        logger.info(
            "No specific file targeted, scanning directories for all media files..."
        )
        for directory in inputs.dirs:
            dir_path = Path(directory)
            if dir_path.is_dir():
                media_files.extend(get_audio(dir_path))
            else:
                logger.warning(f"❗ Skipping invalid directory: {dir_path}")

    if not media_files:
        logger.warning("🤷 No media files found to process.")
        return

    extracted_subs = extract_all_subtitles(
        files=media_files,
        lang_ext=inputs.lang_ext,
        lang_ext_original=inputs.lang_ext_original,
        overwrite=inputs.overwrite,
        existence_check_lang=inputs.lang_ext,
        strict=True,
    )
    if not extracted_subs:
        logger.info("🤔 No new subtitles were extracted.")
        return

    if not inputs.verify:
        return

    logger.info("🕵️ Verifying language of newly extracted files...")
    for sub_path in extracted_subs:
        detected_lang = detect_language(sub_path)
        if detected_lang != inputs.lang_ext_original:
            logger.error(
                f"❌ Language mismatch for '{sub_path.name}'! Expected '{inputs.lang_ext_original}', detected '{detected_lang}'. Deleting file."
            )
            sub_path.unlink()
        else:
            logger.success(f"✅ Language verified for '{sub_path.name}'.")
