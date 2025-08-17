# subplz/watcher.py
import time
import sys
import os
import yaml
import json
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from .cli import BatchParams
from .batch import run_batch

def get_host_path(config, path_from_job):
    """Translates a container path to a host path using the path_map."""
    path_map = config.get('watcher', {}).get('path_map', {})
    for docker_path, host_path in path_map.items():
        if path_from_job.startswith(docker_path):
            print(f"   - Translating container path: {path_from_job} -> {host_path}")
            return path_from_job.replace(docker_path, host_path, 1)

    if os.path.isdir(path_from_job):
        print(f"   - Path is already a valid host path: {path_from_job}")
        return path_from_job

    raise FileNotFoundError(f"Could not resolve '{path_from_job}' to a valid host directory.")

def process_job_file(job_file_path, full_config):
    """
    Reads a single job file and triggers the full batch pipeline.
    """
    print(f"\n--- Processing Job: {os.path.basename(job_file_path)} ---")

    try:
        time.sleep(1)
        with open(job_file_path, 'r', encoding='utf-8') as f:
            job_data = json.load(f)

        path_from_job = job_data.get('directory')
        if not path_from_job:
            raise ValueError("Job file is missing the required 'directory' key.")

        host_target_dir = get_host_path(full_config, path_from_job)
        print(f"🚀 Triggering full batch pipeline for: {host_target_dir}")

        pipeline = full_config.get('batch_pipeline')
        if not pipeline:
            raise ValueError("Missing 'batch_pipeline' in config file.")

        batch_inputs = BatchParams(
            subcommand='batch',
            dirs=[host_target_dir],
            pipeline=pipeline,
            config=None
        )

        run_batch(batch_inputs)
        print("✅ Batch pipeline processing complete for job.")

        os.remove(job_file_path)
        print(f"🗑️ Deleted successful job file: {os.path.basename(job_file_path)}")

    except Exception as e:
        print(f"❌ Error processing job {os.path.basename(job_file_path)}: {e}")

        # --- IMPROVED ERROR HANDLING LOGIC ---
        watcher_settings = full_config.get('watcher', {})
        error_dir_path = watcher_settings.get('error_directory')

        if not error_dir_path:
            # This case handles when the key is missing from the config entirely.
            print("⚠️  'error_directory' key not found in config. Failed job file was not moved.")
            return # Exit the function

        if not os.path.isdir(error_dir_path):
            # This case handles when the path exists but isn't a directory, or doesn't exist at all.
            print(f"⚠️  The specified error_directory '{error_dir_path}' does not exist or is not a directory. Failed job file was not moved.")
            return # Exit the function

        # If we get here, the directory is valid, so we can try to move the file.
        try:
            error_path = os.path.join(error_dir_path, os.path.basename(job_file_path))
            os.rename(job_file_path, error_path)
            print(f"🗄️ Moved failed job to: {error_path}")
        except Exception as move_e:
            print(f"❌ CRITICAL: Additionally failed to move the job file to the error directory: {move_e}")


class JobEventHandler(FileSystemEventHandler):
    def __init__(self, full_config):
        self.full_config = full_config
        # print("✅ Watcher event handler initialized.")

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.json'):
            return
        process_job_file(event.src_path, self.full_config)

def run_watcher(args):
    """Main function to start the watcher, called by the CLI."""
    print(f"Starting watcher with config: {args.config}")
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing YAML file: {e}")
        sys.exit(1)

    watcher_settings = config.get('watcher', {})
    job_dir = watcher_settings.get('jobs')

    if not job_dir or not os.path.isdir(job_dir):
        print(f"FATAL: 'watcher.jobs' directory not found or not specified in config: {job_dir}")
        sys.exit(1)

    print("\n--- Checking for pre-existing jobs... ---")
    try:
        existing_jobs = [f for f in os.listdir(job_dir) if f.endswith('.json')]
        if not existing_jobs:
            print("No pre-existing jobs found.")
        else:
            print(f"Found {len(existing_jobs)} pre-existing job(s). Processing now...")
            for job_filename in sorted(existing_jobs):
                job_filepath = os.path.join(job_dir, job_filename)
                process_job_file(job_filepath, config)
    except Exception as e:
        print(f"❌ An error occurred during the initial scan of existing jobs: {e}")
    print("--- Initial scan complete. ---\n")

    print(f"👀 Watching for .json files in: {job_dir}")

    event_handler = JobEventHandler(full_config=config)
    poll_interval = watcher_settings.get('polling_interval_seconds')

    if poll_interval is not None:
        if not isinstance(poll_interval, (int, float)) or poll_interval <= 0:
            print(f"FATAL: 'polling_interval_seconds' in config must be a positive number, but found: {poll_interval}")
            sys.exit(1)
        print(f"⚠️  Polling observer enabled with a {poll_interval}-second interval (required for WSL/network drives).")
        observer = PollingObserver(timeout=poll_interval)
    else:
        print("✅ Using native OS event observer")
        observer = Observer()

    observer.schedule(event_handler, job_dir, recursive=False)
    observer.start()
    print("✅ Watcher is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    print("\nWatcher stopped.")