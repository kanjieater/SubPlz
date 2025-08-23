import os
import sys
import json
import time
import re
import hashlib
from pathlib import Path
from ..logger import logger
from ..files import VIDEO_FORMATS, AUDIO_FORMATS, get_true_stem
from ..utils import get_host_path, get_docker_path


def create_job_file(job_dir, media_dir_path, episode_path, full_config):
    """
    Creates a deterministic, human-readable, and length-safe JSON job file.
    """
    base_name = Path(episode_path).stem
    # Replace characters that are invalid in filenames on most operating systems.
    sanitized_base_name = re.sub(r'[<>:"/\\|?*]', "_", base_name)
    # 2. Create a short, unique hash from the FULL path to prevent collisions
    #    between identically named files in different folders.
    media_path_str = str(episode_path)
    full_hash = hashlib.md5(media_path_str.encode("utf-8")).hexdigest()
    short_hash = full_hash[
        :8
    ]  # Take the first 8 characters for a short but effective ID.
    # 3. Combine the parts and enforce a reasonable max filename length (e.g., 240 chars).
    prefix = "scanner_"
    suffix = f"_{short_hash}.json"
    # Calculate the max length allowed for the human-readable part.
    # 255 is a common limit, we'll use 240 to be safe.
    max_base_name_len = 240 - len(prefix) - len(suffix)
    truncated_base_name = sanitized_base_name[:max_base_name_len]
    job_filename = f"{prefix}{truncated_base_name}{suffix}"
    job_filepath = os.path.join(job_dir, job_filename)
    docker_media_dir = get_docker_path(full_config, media_dir_path)
    docker_episode_path = get_docker_path(full_config, episode_path)

    job_data = {
        "directory": docker_media_dir,
        "episode_path": docker_episode_path,
        "source": "library_scanner",
    }

    with open(job_filepath, "w", encoding="utf-8") as f:
        json.dump(job_data, f, indent=2)

    logger.success(f"Created/Updated job for file: {Path(episode_path).name}")


def is_blacklisted(filename, blacklist_terms):
    """
    Check if filename should be blacklisted using smarter word boundary matching.
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
    watcher_settings = config.get("watcher", {})

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
    files_scanned = 0

    if target_file:
        host_target_file = get_host_path(config, target_file)
        target_path = Path(host_target_file)
        if not target_path.is_file():
            logger.error(
                f"Target file '{host_target_file}' does not exist. Aborting scan."
            )
            return

        logger.info(f"Focusing scan on single file: {target_path.name}")
        files_scanned = 1
        if check_file_for_missing_subs(
            str(target_path.parent), target_path.name, scanner_settings
        ):
            create_job_file(job_dir, str(target_path.parent), target_path, config)
            logger.success("--- Scan Complete. Scanned 1 file, created 1 new job. ---")
        else:
            logger.success("--- Scan Complete. Scanned 1 file, no job needed. ---")
        return

    jobs_to_create = []

    if override_dirs:
        logger.info(f"Scanning directories provided via command line: {override_dirs}")
        content_dirs = [get_host_path(config, d) for d in override_dirs]
    else:
        logger.info("Scanning directories from config file's watcher.path_map")
        path_map = watcher_settings.get("path_map", {})
        logger.debug(f"Path map from config: {path_map}")
        content_dirs = list(path_map.values())

    logger.info(f"Content directories to scan: {content_dirs}")

    if not content_dirs:
        logger.error("No content directories to scan.")
        return

    blacklist_dirs = scanner_settings.get("blacklist_dirs", [])
    logger.debug(f"Blacklisted directories: {blacklist_dirs}")

    for content_dir in content_dirs:
        logger.info(f"Finding files in: {content_dir}...")
        if not os.path.isdir(content_dir):
            logger.warning(f"Directory not found, skipping: {content_dir}")
            continue

        for root, dirs, files in os.walk(content_dir):
            dirs[:] = [d for d in dirs if d not in blacklist_dirs]

            for file_name in files:
                files_scanned += 1
                if check_file_for_missing_subs(root, file_name, scanner_settings):
                    episode_path = Path(root) / file_name
                    jobs_to_create.append(episode_path)

    if not jobs_to_create:
        logger.success(
            f"--- Scan Complete. Scanned {files_scanned} files, no new jobs needed. ---"
        )
        return

    logger.info(
        f"Found {len(jobs_to_create)} files needing jobs. Sorting and creating job files..."
    )

    sorted_paths = sorted(jobs_to_create)

    job_counter = 0
    for episode_path in sorted_paths:
        create_job_file(job_dir, episode_path.parent, episode_path, config)
        job_counter += 1
        time.sleep(0.1)  # Guarantees sequential creation timestamps

    logger.success(
        f"--- Scan Complete. Scanned {files_scanned} files, created {job_counter} new jobs. ---"
    )


def run_scanner(args):
    """Main entry point for the scanner command."""
    config = args.config_data
    override_dirs = args.dirs
    target_file = args.file
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
