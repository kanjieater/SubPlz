# automation/scanner.py
import os
import sys
import json
import time
from .config import load_config

# Assuming VIDEO_FORMATS is defined elsewhere
VIDEO_FORMATS = ['mkv', 'mp4', 'avi']

def create_job_file(job_dir, media_dir_path):
    # ... (this function remains the same)
    timestamp = int(time.time() * 1000)
    safe_basename = os.path.basename(media_dir_path).replace(" ", "_")
    job_filename = f"scanner_{safe_basename}_{timestamp}.json"
    job_filepath = os.path.join(job_dir, job_filename)

    job_data = {
        "media_directory": media_dir_path,
        "source": "library_scanner",
        "task": "ensure_all_subs_exist"
    }

    with open(job_filepath, 'w', encoding='utf-8') as f:
        json.dump(job_data, f, indent=2)

    print(f"  ✅ Created job for directory: {os.path.basename(media_dir_path)}")


def scan_library(config):
    """Scans the library for videos missing ANY of the target subtitles."""
    settings = config['scanner']
    content_dirs = config.get('consumer', {}).get('path_map', {}).values()
    job_dir = config['consumer']['jobs']

    print("\n--- Starting Library Scan ---")

    # --- KEY CHANGE: Read a list of extensions ---
    target_exts = settings.get('target_sub_extensions', [])
    if not target_exts:
        print("Warning: 'target_sub_extensions' is empty in config. Nothing to scan for.")
        return

    print(f"Checking for {len(target_exts)} required subtitle extensions: {', '.join(target_exts)}")

    blacklist_files = [fn.lower() for fn in settings.get('blacklist_filenames', [])]
    blacklist_dirs = settings.get('blacklist_dirs', [])
    video_exts = ['.' + ext.lower() for ext in VIDEO_FORMATS]

    job_counter = 0
    # --- KEY CHANGE: Keep track of directories we've already created jobs for ---
    jobs_created_for_dir = set()

    for content_dir in content_dirs:
        print(f"\nScanning directory: {content_dir}...")
        if not os.path.isdir(content_dir):
            print(f"  ⚠️ Warning: Directory not found, skipping: {content_dir}")
            continue

        for root, dirs, files in os.walk(content_dir):
            # Stop processing this directory if a job has already been created for it
            if root in jobs_created_for_dir:
                dirs[:] = [] # Prune subdirectories from os.walk
                continue

            dirs[:] = [d for d in dirs if d not in blacklist_dirs]

            for file in files:
                if any(bl_word in file.lower() for bl_word in blacklist_files):
                    continue

                file_basename, file_ext = os.path.splitext(file)
                if file_ext.lower() in video_exts:
                    # --- KEY CHANGE: Loop through all required extensions ---
                    for expected_ext in target_exts:
                        expected_sub_path = os.path.join(root, file_basename + expected_ext)

                        if not os.path.exists(expected_sub_path):
                            print(f"  - Missing '{expected_ext}' for: {file}")

                            # Create a job for the directory, but only once per scan
                            if root not in jobs_created_for_dir:
                                create_job_file(job_dir, root)
                                jobs_created_for_dir.add(root)
                                job_counter += 1

                            # Once we find one missing sub, we can stop checking for this directory
                            break # Exit the inner loop (for expected_ext)

    print(f"\n--- Scan Complete. Created {job_counter} new jobs. ---")

def main():
    """Main function to be called by the script entry point."""
    if len(sys.argv) < 2:
        print("Usage: python ./automation/scanner.py <path_to_config.yml>")
        sys.exit(1)

    try:
        config_file_path = sys.argv[1]
        config = load_config(config_file_path)
        scan_library(config)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()