import time
import sys
import os
import json
import threading
import shutil
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from ..logger import logger, configure_logging
from ..cli import BatchParams
from ..batch import run_batch
from ..utils import get_host_path


def process_job_file(job_file_path, full_config):
    """
    Directly calls the batch processing function and handles the outcome.
    """
    logger.info(f"--- Starting Job: {os.path.basename(job_file_path)} ---")
    job_succeeded = False
    try:
        # Create the inputs for run_batch from the job file
        with open(job_file_path, "r", encoding="utf-8") as f:
            job_data = json.load(f)

        host_target_dir = get_host_path(full_config, job_data["directory"])
        host_episode_path = (
            get_host_path(full_config, job_data["episode_path"])
            if job_data.get("episode_path")
            else None
        )

        # Construct the BatchParams object that run_batch expects
        batch_inputs = BatchParams(
            subcommand="batch",
            dirs=[host_target_dir],
            file=host_episode_path,
            pipeline=full_config.get("batch_pipeline"),
            config=None,
            config_data=full_config,
        )

        run_batch(batch_inputs)

        # If run_batch completes without raising an exception, it succeeded
        job_succeeded = True

    except Exception:
        # run_batch raises a generic exception on failure.
        # The detailed error is already logged inside batch.py
        logger.error(
            f"Job '{os.path.basename(job_file_path)}' failed because one or more pipeline steps failed."
        )
        job_succeeded = False

    # --- File Cleanup ---
    if job_succeeded:
        logger.success(
            f"Successfully completed job '{os.path.basename(job_file_path)}'."
        )
        try:
            os.remove(job_file_path)
            logger.info(
                f"🗑️ Deleted successful job file: {os.path.basename(job_file_path)}"
            )
        except FileNotFoundError:
            logger.debug(
                f"Job file {os.path.basename(job_file_path)} was already deleted."
            )
        except OSError as e:
            logger.error(f"Failed to delete successful job file {job_file_path}: {e}")
    else:
        logger.info(
            f"Moving failed job '{os.path.basename(job_file_path)}' to error directory."
        )
        base_dirs = full_config.get("base_dirs", {})
        error_dir_path = base_dirs.get("watcher_errors")
        if not error_dir_path:
            logger.warning(
                "'base_dirs.watcher_errors' key not found in config. Failed job file was not moved."
            )
            return
        try:
            if os.path.exists(job_file_path):
                error_path = os.path.join(
                    error_dir_path, os.path.basename(job_file_path)
                )
                shutil.move(job_file_path, error_path)
                logger.info(f"🗄️ Moved failed job to: {error_path}")
            else:
                logger.warning(
                    f"Failed job file {os.path.basename(job_file_path)} was already gone. Skipping move."
                )
        except Exception:
            logger.opt(exception=True).critical(
                "Additionally failed to move the job file to the error directory!"
            )


def get_next_job(job_dir):
    """Get the oldest job file from the directory, or None if no jobs exist."""
    try:
        job_paths = [
            os.path.join(job_dir, f) for f in os.listdir(job_dir) if f.endswith(".json")
        ]
        if not job_paths:
            return None
        logger.info(f"📊 Found {len(job_paths)} job(s) in the queue.")
        oldest_job = min(job_paths, key=os.path.getctime)
        return oldest_job
    except Exception as e:
        logger.error(f"Error scanning for next job: {e}")
        return None


def process_job_queue(job_dir, full_config):
    """Process all jobs in the directory, one at a time, in creation order."""
    while True:
        next_job = get_next_job(job_dir)
        if next_job is None:
            logger.info("No more jobs to process in the current queue.")
            break

        process_job_file(next_job, full_config)
        time.sleep(0.5)


class JobEventHandler(FileSystemEventHandler):
    def __init__(self, full_config, job_dir):
        self.full_config = full_config
        self.job_dir = job_dir
        self.processing_lock = threading.Lock()
        logger.success("Watcher event handler initialized.")

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".json"):
            return
        if not os.path.exists(event.src_path):
            logger.debug(f"Already Processed: {os.path.basename(event.src_path)}")
            return
        logger.info(f"New job detected: {os.path.basename(event.src_path)}")

        if self.processing_lock.acquire(blocking=False):
            logger.info("Acquired lock. Starting job queue processing...")
            try:
                process_job_queue(self.job_dir, self.full_config)
            finally:
                logger.info("...job queue processing finished. Releasing lock.")
                self.processing_lock.release()
        else:
            logger.info(
                "Job queue is already being processed. Ignoring redundant event."
            )


def run_watcher(args):
    """Main function to start the watcher, called by the CLI."""
    # The main watcher process configures its own logger
    config = args.config_data
    observer = None

    try:
        logger.info(f"Starting watcher with config file: {args.config}")
        base_dirs = config.get("base_dirs", {})
        job_dir = base_dirs.get("watcher_jobs")
        watcher_settings = config.get("watcher", {})

        if not job_dir or not os.path.isdir(job_dir):
            raise RuntimeError(
                f"'base_dirs.watcher_jobs' directory not found or not configured: {job_dir}"
            )

        logger.info("Scanning for existing jobs on startup...")
        process_job_queue(job_dir, config)

        logger.info(f"👀 Watching for .json files in: {job_dir}")
        event_handler = JobEventHandler(full_config=config, job_dir=job_dir)
        poll_interval = watcher_settings.get("polling_interval_seconds")

        if poll_interval is not None:
            if not isinstance(poll_interval, (int, float)) or poll_interval <= 0:
                raise ValueError(
                    f"'polling_interval_seconds' must be a positive number, but found: {poll_interval}"
                )
            logger.warning(
                f"Polling observer enabled with a {poll_interval}-second interval (for WSL/network drives)."
            )
            observer = PollingObserver(timeout=poll_interval)
        else:
            logger.info("Using native OS event observer (efficient).")
            observer = Observer()

        observer.schedule(event_handler, job_dir, recursive=False)
        observer.start()
        logger.success("Watcher is running. Press Ctrl+C to stop.")

        while observer.is_alive():
            observer.join(1)

    except KeyboardInterrupt:
        logger.info("\nWatcher stopped by user (Ctrl+C).")
    except Exception:
        logger.opt(exception=True).critical(
            "A fatal error occurred in the watcher's main loop."
        )
        sys.exit(1)
    finally:
        if observer and observer.is_alive():
            observer.stop()
            observer.join()
