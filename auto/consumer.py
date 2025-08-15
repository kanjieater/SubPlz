# automation/consumer.py
import time
import json
import os
import subprocess
import sys

# Import the centralized config loader from the same 'automation' package
from .config import load_config

def get_host_path(config, path_from_job):
    """
    Determines the correct host path. If the path starts with a known
    container path from the path_map, it translates it. Otherwise, it
    assumes it's already a valid host path.
    """
    path_map = config.get('job_consumer_settings', {}).get('path_map', {})
    for docker_path, host_path in path_map.items():
        if path_from_job.startswith(docker_path):
            # It's a container path, translate it and return
            print(f"   - Translating container path: {path_from_job}")
            return path_from_job.replace(docker_path, host_path, 1)

    # If no match was found, assume it's already a valid host path
    if os.path.isdir(path_from_job):
        print(f"   - Path is already a valid host path: {path_from_job}")
        return path_from_job

    # If we get here, the path is neither a translatable container path nor an existing host path
    print(f"⚠️ Warning: Could not resolve '{path_from_job}' to a valid host directory.")
    return None

def process_job_file(full_config, job_file_path):
    """Reads a single job file, runs SubPlz, and cleans up the file."""
    print(f"--- Processing Job: {os.path.basename(job_file_path)} (PID: {os.getpid()}) ---")

    # Use only the settings relevant to the consumer
    config = full_config['job_consumer_settings']

    try:
        # Wait a brief moment to ensure the file is fully written by the producer
        time.sleep(1)
        with open(job_file_path, 'r', encoding='utf-8') as f:
            job_data = json.load(f)

        print(f"  Job Data: {json.dumps(job_data, indent=2)}")

        path_from_job = job_data.get('media_directory')
        if not path_from_job:
            raise ValueError("Job file is missing 'media_directory' key.")

        host_target_dir = get_host_path(full_config, path_from_job)
        if not host_target_dir:
            raise ValueError(f"Failed to resolve a valid host path for '{path_from_job}'")

        print(f"🚀 Triggering SubPlz for host directory: {host_target_dir}")

        # Build the command arguments, replacing the placeholder
        args = [config['subplz_command']]
        for arg in config['subplz_args']:
            args.append(arg.replace("{directory}", host_target_dir))

        print(f"  Executing command: {' '.join(args)}")
        result = subprocess.run(args, check=True, capture_output=True, text=True, encoding='utf-8')

        if result.stdout:
            print(f"  STDOUT: {result.stdout.strip()}")
        if result.stderr:
            print(f"  STDERR: {result.stderr.strip()}")
        print(f"✅ SubPlz processing complete.")

    except Exception as e:
        print(f"❌ Error processing job {job_file_path}: {e}")
        error_dir = config.get('error_directory')
        if error_dir and os.path.isdir(error_dir):
            try:
                error_path = os.path.join(error_dir, os.path.basename(job_file_path))
                os.rename(job_file_path, error_path)
                print(f"Moved failed job to: {error_path}")
            except Exception as move_e:
                print(f"❌ Additionally failed to move job file: {move_e}")
        # Exit with an error code to signal failure
        sys.exit(1)

    try:
        os.remove(job_file_path)
        print(f"🗑️ Deleted job file: {os.path.basename(job_file_path)}")
    except Exception as e:
        print(f"❌ Failed to delete job file {job_file_path}: {e}")

def main():
    """Main entry point for the consumer script."""
    if len(sys.argv) < 3:
        print("Usage: python consumer.py <path_to_config.yml> <path_to_job_file.json>")
        sys.exit(1)

    try:
        config_file_path = sys.argv[1]
        job_file_path = sys.argv[2]

        # Use the centralized loader
        full_config = load_config(config_file_path)
        process_job_file(full_config, job_file_path)

    except Exception as e:
        print(f"An unhandled error occurred in consumer process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()