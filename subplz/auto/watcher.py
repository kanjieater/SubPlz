import time
import sys
import os
import json
import threading
import multiprocessing
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from ..logger import logger, configure_logging
from ..cli import BatchParams
from ..batch import run_batch


def _run_job_worker(job_file_path, full_config):
    """
    The job logic that runs in a separate process. Exits with 0 on success, 1 on failure.
    """
    try:
        configure_logging(full_config)

        logger.info(f"--- [PID:{os.getpid()}] Processing Job: {os.path.basename(job_file_path)} ---")
        time.sleep(1)
        with open(job_file_path, "r", encoding="utf-8") as f:
            job_data = json.load(f)

        path_from_job = job_data.get("directory")
        if not path_from_job:
            raise ValueError("Job file is missing the required 'directory' key.")

        episode_path_from_job = job_data.get("episode_path")
        host_target_dir = get_host_path(full_config, path_from_job)
        host_episode_path = None
        if episode_path_from_job:
            host_episode_path = get_host_path(full_config, episode_path_from_job)

        logger.info(f"🚀 Triggering batch pipeline for dir: {host_target_dir}")
        if host_episode_path:
            logger.info(f"      Focused on file: {Path(host_episode_path).name}")

        pipeline = full_config.get("batch_pipeline")
        if not pipeline:
            raise ValueError("Missing 'batch_pipeline' in config file.")

        batch_inputs = BatchParams(
            subcommand="batch",
            dirs=[host_target_dir],
            file=host_episode_path,
            pipeline=pipeline,
            config=None,
            config_data=full_config,
        )

        run_batch(batch_inputs)
        sys.exit(0)  # Signal success
    except Exception as e:
        logger.opt(exception=True).error(
            f"A fatal error occurred in the job sub-process for {os.path.basename(job_file_path)}: {e}"
        )
        sys.exit(1)  # Signal failure

def _spawn_and_wait_for_job(job_file_path, full_config):
    """
    Creates, starts, and waits for the job worker process to complete.
    Returns the process exit code.
    """
    logger.info(f"--- Spawning new process for Job: {os.path.basename(job_file_path)} ---")
    ctx = multiprocessing.get_context('spawn')
    process = ctx.Process(target=_run_job_worker, args=(job_file_path, full_config))
    process.start()
    process.join()
    return process.exitcode

def _handle_job_completion(exitcode, job_file_path, full_config):
    """
    Deletes or moves the job file based on the worker process's exit code.
    """
    if exitcode == 0:
        logger.success(f"Batch pipeline processing complete for job '{os.path.basename(job_file_path)}'.")
        try:
            os.remove(job_file_path)
            logger.info(f"🗑️ Deleted successful job file: {os.path.basename(job_file_path)}")
        except OSError as e:
            logger.error(f"Failed to delete successful job file {job_file_path}: {e}")
    else:
        logger.error(f"Job process for '{os.path.basename(job_file_path)}' failed with exit code {exitcode}.")
        base_dirs = full_config.get("base_dirs", {})
        error_dir_path = base_dirs.get("watcher_errors")
        if not error_dir_path:
            logger.warning("'base_dirs.watcher_errors' key not found in config. Failed job file was not moved.")
            return
        try:
            error_path = os.path.join(error_dir_path, os.path.basename(job_file_path))
            os.rename(job_file_path, error_path)
            logger.info(f"🗄️ Moved failed job to: {error_path}")
        except Exception:
            logger.opt(exception=True).critical("Additionally failed to move the job file to the error directory!")

def process_job_file(job_file_path, full_config):
    """
    Orchestrates running a job in a separate process and handling the result.
    """
    exitcode = _spawn_and_wait_for_job(job_file_path, full_config)
    _handle_job_completion(exitcode, job_file_path, full_config)


def get_host_path(config, path_from_job):
    """
    Translates a container path to a host path, or confirms a host path's existence.
    """
    path_map = config.get("watcher", {}).get("path_map", {})
    for docker_path, host_path in path_map.items():
        if path_from_job.startswith(docker_path):
            logger.debug(f"Translating container path: {path_from_job} -> {host_path}")
            return path_from_job.replace(docker_path, host_path, 1)

    if os.path.exists(path_from_job):
        logger.debug(f"Path is already a valid host path: {path_from_job}")
        return path_from_job

    raise FileNotFoundError(
        f"Could not resolve '{path_from_job}' to a valid host path."
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