import os
import sys
import json
import time
from ..logger import logger

# --- KEY CHANGE #1: Import the centralized format lists ---
from ..files import VIDEO_FORMATS, AUDIO_FORMATS

def create_job_file(job_dir, media_dir_path):
    timestamp = int(time.time() * 1000)
    safe_basename = os.path.basename(media_dir_path).replace(" ", "_")
    job_filename = f"scanner_{safe_basename}_{timestamp}.json"
    job_filepath = os.path.join(job_dir, job_filename)

    # Standardize on the 'directory' key
    job_data = {
        "directory": media_dir_path,
        "source": "library_scanner"
    }

    with open(job_filepath, 'w', encoding='utf-8') as f:
        json.dump(job_data, f, indent=2)

    logger.success(f"Created job for directory: {os.path.basename(media_dir_path)}")


def scan_library(config):
    """Scans the library for videos missing ANY of the target subtitles."""
    # Use the logger instead of print
    scanner_settings = config.get('scanner', {})
    watcher_settings = config.get('watcher', {})

    # The scanner needs to know where the media directories are and where to put the jobs.
    content_dirs = [v for k, v in watcher_settings.get('path_map', {}).items()]
    job_dir = watcher_settings.get('jobs')

    if not content_dirs:
        logger.error("No 'path_map' directories found in watcher settings. Cannot scan library.")
        return
    if not job_dir:
        logger.error("No 'jobs' directory found in watcher settings. Cannot create jobs.")
        return

    logger.info("--- Starting Library Scan ---")
    target_exts = scanner_settings.get('target_sub_extensions', [])
    if not target_exts:
        logger.warning("'target_sub_extensions' is empty in config. Nothing to scan for.")
        return

    logger.info(f"Checking for {len(target_exts)} required subtitle extensions: {', '.join(target_exts)}")

    blacklist_files = [fn.lower() for fn in scanner_settings.get('blacklist_filenames', [])]
    blacklist_dirs = scanner_settings.get('blacklist_dirs', [])

    # --- KEY CHANGE #2: Combine both lists for comprehensive media detection ---
    media_exts = ['.' + ext.lower() for ext in VIDEO_FORMATS + AUDIO_FORMATS]

    job_counter = 0
    jobs_created_for_dir = set()

    for content_dir in content_dirs:
        logger.info(f"Scanning directory: {content_dir}...")
        if not os.path.isdir(content_dir):
            logger.warning(f"Directory not found, skipping: {content_dir}")
            continue

        for root, dirs, files in os.walk(content_dir):
            if root in jobs_created_for_dir:
                dirs[:] = []
                continue

            dirs[:] = [d for d in dirs if d not in blacklist_dirs]

            for file in files:
                if any(bl_word in file.lower() for bl_word in blacklist_files):
                    continue

                file_basename, file_ext = os.path.splitext(file)
                # --- KEY CHANGE #3: Check against the new combined list ---
                if file_ext.lower() in media_exts:
                    for expected_ext in target_exts:
                        expected_sub_path = os.path.join(root, file_basename + expected_ext)
                        if not os.path.exists(expected_sub_path):
                            logger.info(f"Missing '{expected_ext}' for media file: {file}")
                            if root not in jobs_created_for_dir:
                                create_job_file(job_dir, root)
                                jobs_created_for_dir.add(root)
                                job_counter += 1
                            break

    logger.success(f"--- Scan Complete. Created {job_counter} new jobs. ---")


def run_scanner(args):
    """Main entry point for the scanner command."""
    config = args.config_data

    logger.info("Scanner command initiated.")
    try:
        scan_library(config)
        logger.info("Scanner command finished successfully.")
    except Exception:
        logger.opt(exception=True).critical("A fatal error occurred during the library scan.")
        sys.exit(1)