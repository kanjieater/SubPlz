# automation/scanner.py
import os
import sys
import json
import time
from subplz.files import VIDEO_FORMATS
from .config import load_config

def create_job_file(job_dir, media_dir_path):
    """Creates a new .json job file in the queue directory."""
    timestamp = int(time.time() * 1000)
    safe_basename = os.path.basename(media_dir_path).replace(" ", "_")
    job_filename = f"scanner_{safe_basename}_{timestamp}.json"
    job_filepath = os.path.join(job_dir, job_filename)

    job_data = {
        "media_directory": media_dir_path,
        "source": "library_scanner",
        "task": "generate_missing_sub"
    }

    with open(job_filepath, 'w', encoding='utf-8') as f:
        json.dump(job_data, f, indent=2)

    print(f"  ✅ Created job file: {job_filename}")


def scan_library(config):
    """Scans the library for videos missing target subtitles and creates jobs."""
    settings = config['scanner_settings']
    job_dir = config['job_consumer_settings']['job_directory']

    print("\n--- Starting Library Scan ---")

    content_dirs = settings.get('content_dirs', [])
    target_ext = settings.get('target_sub_extension', '.az.srt')
    blacklist_files = [fn.lower() for fn in settings.get('blacklist_filenames', [])]
    blacklist_dirs = settings.get('blacklist_dirs', [])

    video_exts = ['.' + ext.lower() for ext in VIDEO_FORMATS]
    print(f"   Using {len(video_exts)} video formats imported from subplz.files")

    job_counter = 0

    for content_dir in content_dirs:
        print(f"\nScanning directory: {content_dir}...")
        for root, dirs, files in os.walk(content_dir):
            dirs[:] = [d for d in dirs if d not in blacklist_dirs]

            for file in files:
                if any(bl_word in file.lower() for bl_word in blacklist_files):
                    continue

                file_basename, file_ext = os.path.splitext(file)
                if file_ext.lower() in video_exts:
                    expected_sub_path = os.path.join(root, file_basename + target_ext)

                    if not os.path.exists(expected_sub_path):
                        print(f"  - Missing '{target_ext}' for: {file}")
                        media_directory_path = os.path.dirname(os.path.join(root, file))
                        create_job_file(job_dir, media_directory_path)
                        job_counter += 1

    print(f"\n--- Scan Complete. Created {job_counter} new jobs. ---")


def main():
    """Main function to be called by the script entry point."""
    if len(sys.argv) < 2:
        print("Usage: subplz-scanner <path_to_config.yml>")
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