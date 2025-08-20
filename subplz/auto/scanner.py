import os
import sys
import json
import time
import re
from pathlib import Path
from ..logger import logger
from ..files import VIDEO_FORMATS, AUDIO_FORMATS, get_true_stem


def create_job_file(job_dir, media_dir_path, episode_path):
    """Creates a JSON job file for the watcher to process."""
    safe_basename = Path(episode_path).stem.replace(" ", "_")
    timestamp = int(time.time() * 1000)
    job_filename = f"scanner_{safe_basename}_{timestamp}.json"
    job_filepath = os.path.join(job_dir, job_filename)

    job_data = {
        "directory": str(media_dir_path),
        "episode_path": str(episode_path),
        "source": "library_scanner",
    }

    with open(job_filepath, "w", encoding="utf-8") as f:
        json.dump(job_data, f, indent=2)

    logger.success(f"Created job for file: {Path(episode_path).name}")


def is_blacklisted(filename, blacklist_terms):
    """
    FIXED: Check if filename should be blacklisted using smarter word boundary matching.

    This prevents false positives like "OP" in "[Opus 2.0]"
    """
    if not blacklist_terms:
        return False

    filename_lower = filename.lower()

    for term in blacklist_terms:
        term_lower = term.lower()

        # Use word boundary regex for better matching
        # This will match "OP" as a whole word but not "OP" inside "Opus"
        pattern = r"\b" + re.escape(term_lower) + r"\b"

        if re.search(pattern, filename_lower):
            logger.debug(
                f"File '{filename}' matched blacklist term '{term}' (word boundary match)"
            )
            return True

    return False


def check_file_for_missing_subs(root, file_name, scanner_settings):
    """
    Checks if a media file is missing ANY of its target subtitles.
    Returns True if at least one required sub is missing, triggering a job. False otherwise.
    """
    target_exts = scanner_settings.get("target_sub_extensions", [])
    blacklist_files = scanner_settings.get("blacklist_filenames", [])
    media_exts = ["." + ext.lower() for ext in VIDEO_FORMATS + AUDIO_FORMATS]

    logger.debug(f"Scanner checking: {file_name}")

    # FIXED: Use smarter blacklist checking
    if is_blacklisted(file_name, blacklist_files):
        logger.debug(f"File '{file_name}' is blacklisted, skipping")
        return False

    media_path = Path(root) / file_name

    # Check if target_exts is empty or None
    if not target_exts:
        logger.warning("No target subtitle extensions configured! Check your config.")
        return False

    if media_path.suffix.lower() in media_exts:
        logger.info(f"-> Found media file: {media_path.name}")

        # Use get_true_stem to handle complex filenames
        file_basename = get_true_stem(media_path)

        logger.info(f"--> Base name identified as: '{file_basename}'")

        # Track what we're looking for vs what we found
        missing_subs = []
        found_subs = []

        for expected_ext in target_exts:
            # Construct the full path for the expected subtitle file
            expected_sub_path = media_path.parent / (file_basename + expected_ext)

            logger.info(f"---> Checking for subtitle: {expected_sub_path.name}")
            logger.debug(f"     Full path: {expected_sub_path}")

            if not os.path.exists(expected_sub_path):
                logger.info(f"----> MISSING: {expected_sub_path.name}")
                missing_subs.append(expected_ext)
            else:
                logger.info(f"----> FOUND: {expected_sub_path.name}")
                found_subs.append(expected_ext)

        # Summary logging
        if missing_subs:
            logger.info(
                f"--> Missing subtitles: {missing_subs}. Creating job for {media_path.name}"
            )
            return True
        else:
            logger.info(
                f"--> All required subtitles found for {media_path.name}. No job needed."
            )
            return False
    else:
        logger.debug(
            f"File '{file_name}' is not a media file (extension: {media_path.suffix})"
        )
        return False


def scan_library(config, override_dirs=None, target_file=None):
    """Scans the library and creates one job per media file missing subtitles."""
    scanner_settings = config.get("scanner", {})

    base_dirs = config.get("base_dirs", {})
    job_dir = base_dirs.get("watcher_jobs")

    # We still need watcher_settings for non-path related keys like path_map
    watcher_settings = config.get("watcher", {})
    job_dir = watcher_settings.get("jobs")

    # DEBUG: Log the configuration
    logger.debug(f"Scanner settings: {json.dumps(scanner_settings, indent=2)}")
    logger.debug(f"Jobs directory: {job_dir}")

    if not job_dir:
        logger.error(
            "No 'jobs' directory found in watcher settings. Cannot create jobs."
        )
        return

    # Check if job directory exists and is writable
    if not os.path.exists(job_dir):
        logger.error(f"Job directory does not exist: {job_dir}")
        return

    if not os.access(job_dir, os.W_OK):
        logger.error(f"Job directory is not writable: {job_dir}")
        return

    logger.info("--- Starting Library Scan ---")

    job_counter = 0
    files_scanned = 0

    if target_file:
        target_path = Path(target_file)
        if not target_path.is_file():
            logger.error(f"Target file '{target_file}' does not exist. Aborting scan.")
            return

        logger.info(f"Focusing scan on single file: {target_path.name}")
        files_scanned = 1
        if check_file_for_missing_subs(
            str(target_path.parent), target_path.name, scanner_settings
        ):
            create_job_file(job_dir, str(target_path.parent), target_path)
            job_counter += 1
    else:
        if override_dirs:
            logger.info(
                f"Scanning directories provided via command line: {override_dirs}"
            )
            content_dirs = override_dirs
        else:
            logger.info("Scanning directories from config file's watcher.path_map")
            path_map = watcher_settings.get("path_map", {})
            logger.debug(f"Path map from config: {path_map}")
            content_dirs = [v for k, v in path_map.items()]

        logger.info(f"Content directories to scan: {content_dirs}")

        if not content_dirs:
            logger.error("No content directories to scan.")
            return

        blacklist_dirs = scanner_settings.get("blacklist_dirs", [])
        logger.debug(f"Blacklisted directories: {blacklist_dirs}")

        for content_dir in content_dirs:
            logger.info(f"Scanning directory: {content_dir}...")
            if not os.path.isdir(content_dir):
                logger.warning(f"Directory not found, skipping: {content_dir}")
                continue

            for root, dirs, files in os.walk(content_dir):
                # Filter out blacklisted directories
                original_dirs = dirs[:]
                dirs[:] = [d for d in dirs if d not in blacklist_dirs]
                if len(dirs) != len(original_dirs):
                    filtered_out = set(original_dirs) - set(dirs)
                    logger.debug(
                        f"Filtered out blacklisted directories: {filtered_out}"
                    )

                logger.debug(
                    f"Scanning directory: {root} (contains {len(files)} files)"
                )

                for file_name in files:
                    files_scanned += 1
                    if check_file_for_missing_subs(root, file_name, scanner_settings):
                        episode_path = Path(root) / file_name
                        create_job_file(job_dir, root, episode_path)
                        job_counter += 1

    logger.success(
        f"--- Scan Complete. Scanned {files_scanned} files, created {job_counter} new jobs. ---"
    )


def run_scanner(args):
    """Main entry point for the scanner command."""
    config = args.config_data
    override_dirs = args.dirs
    target_file = args.file

    # DEBUG: Log what we received
    logger.debug(f"Scanner args - dirs: {override_dirs}, file: {target_file}")

    logger.info("Scanner command initiated.")
    try:
        scan_library(config, override_dirs=override_dirs, target_file=target_file)
        logger.info("Scanner command finished successfully.")
    except Exception:
        logger.opt(exception=True).critical(
            "A fatal error occurred during the library scan."
        )
        sys.exit(1)
