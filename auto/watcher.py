# automation/watcher.py
import time
import sys
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .config import load_config, ConfigError

class JobEventHandler(FileSystemEventHandler):
    """
    Watches for new files and launches a consumer process for each.
    """
    def __init__(self, config_path, consumer_script_path):
        self.config_path = config_path
        self.consumer_script_path = consumer_script_path
        print("✅ Watcher initialized.")
        print(f"   - Will trigger script: {self.consumer_script_path}")
        print(f"   - Using config: {self.config_path}")

    def on_created(self, event):
        """Called when a file is created in the watched directory."""
        if not event.is_directory and event.src_path.endswith('.json'):
            job_file_path = event.src_path
            print(f"\n--- New Job File Detected: {os.path.basename(job_file_path)} ---")
            print(f"🚀 Launching consumer process...")

            try:
                command = [
                    sys.executable,
                    self.consumer_script_path,
                    self.config_path,
                    job_file_path
                ]
                subprocess.Popen(command)
                print(f"   - Process for {os.path.basename(job_file_path)} started.")
            except Exception as e:
                print(f"❌ Failed to launch consumer process for {job_file_path}: {e}")

def main():
    """Main function to start the watcher."""
    if len(sys.argv) < 2:
        print("Usage: python ./automation/watcher.py <path_to_config.yml>")
        sys.exit(1)

    try:
        config_file_path = sys.argv[1]
        config = load_config(config_file_path)
        # Get the directory to watch using the new keys
        job_dir = config['consumer']['jobs']
    except (ConfigError, KeyError, FileNotFoundError) as e:
        print(f"FATAL: Could not load configuration: {e}")
        sys.exit(1)

    if not os.path.isdir(job_dir):
        print(f"FATAL: Job queue directory specified in config does not exist: {job_dir}")
        sys.exit(1)

    consumer_script = os.path.join(os.path.dirname(__file__), 'consumer.py')
    if not os.path.exists(consumer_script):
        print(f"FATAL: Consumer script not found at: {consumer_script}")
        sys.exit(1)

    print(f"👀 Starting Job Watcher...")
    print(f"   Watching for .json files in: {job_dir}")

    event_handler = JobEventHandler(config_file_path, consumer_script)
    observer = Observer()
    observer.schedule(event_handler, job_dir, recursive=False)
    observer.start()

    print(f"✅ Watcher is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    print("\nWatcher stopped.")

if __name__ == "__main__":
    main()