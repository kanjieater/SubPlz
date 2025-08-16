# automation/consumer.py
import time
import json
import os
import sys

# --- KEY CHANGES START HERE ---
# 1. Import the batch runner and the dataclass it needs
from subplz.batch import run_batch
from subplz.cli import BatchParams
# Note: You may need to adjust the import path for BatchParams depending on your project structure.
# --- KEY CHANGES END HERE ---

from .config import load_config

def get_host_path(config, path_from_job):
    """Translates a container path to a host path using the path_map."""
    path_map = config.get('consumer', {}).get('path_map', {})
    for docker_path, host_path in path_map.items():
        if path_from_job.startswith(docker_path):
            print(f"   - Translating container path: {path_from_job}")
            return path_from_job.replace(docker_path, host_path, 1)

    if os.path.isdir(path_from_job):
        print(f"   - Path is already a valid host path: {path_from_job}")
        return path_from_job

    print(f"⚠️ Warning: Could not resolve '{path_from_job}' to a valid host directory.")
    return None

def process_job_file(full_config, job_file_path):
    """
    Reads a job file and triggers the FULL batch pipeline for the target directory.
    """
    print(f"--- Processing Job via Batch Pipeline: {os.path.basename(job_file_path)} ---")

    try:
        time.sleep(1)
        with open(job_file_path, 'r', encoding='utf-8') as f:
            job_data = json.load(f)

        path_from_job = job_data.get('media_directory')
        if not path_from_job:
            raise ValueError("Job file is missing 'media_directory' key.")

        host_target_dir = get_host_path(full_config, path_from_job)
        if not host_target_dir:
            raise ValueError(f"Failed to resolve a valid host path for '{path_from_job}'")

        print(f"🚀 Triggering full batch pipeline for: {host_target_dir}")

        # --- KEY CHANGES START HERE ---
        # 2. Instead of subprocess, construct the inputs for run_batch
        pipeline = full_config.get('batch_pipeline')
        if not pipeline:
            raise ValueError("Missing 'batch_pipeline' in config file.")

        batch_inputs = BatchParams(
            subcommand='batch',
            dirs=[host_target_dir], # The batch runner expects a list of directories
            pipeline=pipeline,
            config=None # No config file path is needed as we loaded it already
        )

        # 3. Call the same function the manual CLI uses
        run_batch(batch_inputs)
        print(f"✅ Batch pipeline processing complete for job.")
        # --- KEY CHANGES END HERE ---

    except Exception as e:
        # ... (Your existing error handling to move the failed job file) ...
        print(f"❌ Error processing job {job_file_path}: {e}")
        error_dir = full_config.get('consumer', {}).get('error_directory')
        if error_dir and os.path.isdir(error_dir):
            try:
                # ... code to move file ...
                print(f"Moved failed job to error directory.")
            except Exception as move_e:
                print(f"❌ Additionally failed to move job file: {move_e}")
        sys.exit(1)

    try:
        os.remove(job_file_path)
        print(f"🗑️ Deleted job file: {os.path.basename(job_file_path)}")
    except Exception as e:
        print(f"❌ Failed to delete job file {job_file_path}: {e}")

# ... (The rest of your consumer.py 'main' function remains the same) ...