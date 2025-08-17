code
Markdown
# SubPlz Automation Suite

This suite provides a robust, queue-based automation system for processing media files with `SubPlz`. It is a fully integrated part of the `subplz` command-line tool, designed to be triggered by external applications like Bazarr (though any process that can write a json file could trigger it) and to proactively scan your media library for missing subtitles.

## Core Concepts

This system is built on a **Producer/Consumer** model, a powerful and resilient software design pattern.

*   **Producer:** An application that creates a "job" that needs to be done. It places a job file into a designated queue folder.
    *   *Examples:* The **Bazarr Custom Post-Processing** feature and the **`subplz scanner`** command are both Producers.
*   **Queue:** A shared directory that holds job files (in `.json` format). This decouples the Producers from the Consumer, allowing jobs to accumulate safely even if the consumer is offline.
*   **Consumer:** The long-running **`subplz watch`** command. It continuously monitors the queue, picking up one job at a time and processing it through a user-defined pipeline.

## Features

*   **Real-time Processing:** Instantly process media files as soon as a job is created by a producer like Bazarr.
*   **Proactive Library Scanning:** Use the `scanner` command on a schedule (e.g., cron job) to find and queue jobs for media files that are missing subtitles.
*   **Powerful Processing Pipelines:** Define complex, multi-step processing workflows in your `config.yml` that are executed for every job.
*   **Resilient Queue System:** If the watcher is offline, jobs will safely accumulate in the queue and will be processed automatically when it restarts. No events are lost.
*   **Process-Safe Logging:** All components, including the watcher and scanner, log to the same rotating log file, providing a complete, interleaved history of all system activity.
*   **Highly Configurable:** All paths, settings, and the entire processing pipeline are managed in a central `config.yml` file.
*   **Integrated Tooling:** The watcher and scanner are not separate scripts but are first-class commands within the `subplz` CLI.

## Setup and Installation

1.  **Install SubPlz:** Ensure the `subplz` package is installed correctly in your Python environment.
2.  **Create `config.yml`:** Create a `config.yml` file in your project's root directory. Copy the contents from the template below and customize all paths and settings for your system.
3.  **Create Directories:** Manually create the directories you specified in your `config.yml` for `log`, `jobs`, and `error_directory`.

---

## Configuration (`config.yml`)

This single file controls the behavior of all automation commands and the processing pipeline.

```yaml
# ===============================================
# Global Logging Settings
# ===============================================
log:
  # [REQUIRED] The directory where rotating log files will be stored.
  dir: "/mnt/rd/subplz/logs"

# ===============================================
# Settings for the Real-time Job Watcher
# ===============================================
watcher:
  # [REQUIRED] The folder on the HOST machine to watch for new .json job files.
  jobs: "/mnt/rd/subplz/jobs"

  # [OPTIONAL] A directory on the HOST to move job files to if processing fails.
  error_directory: "/mnt/rd/subplz/fails"

  # [REQUIRED] The mapping of paths from inside your Docker containers to your host machine.
  # This allows the script to translate container paths from Bazarr jobs to real host paths.
  path_map:
    # Key: Path inside the container (must end with a slash)
    # Value: Corresponding path on the host machine
    "/data/ja-anime/": "/mnt/an/ja-anime/"
    "/data/test/": "/mnt/g/test/"

  # [OPTIONAL] If this key is present, the watcher uses a polling-based
  # mechanism instead of native OS events. This is less efficient but is
  # REQUIRED if you are running this script inside WSL and watching a
  # directory on the Windows file system (e.g., /mnt/c/).
  # To use the default, efficient watcher, simply remove or comment out this line.
  polling_interval_seconds: 2

# ==========================================================
# Settings for the Scheduled Library Scanner ("Gap-Filler")
# ==========================================================
scanner:
  # [REQUIRED] The subtitle extension to check for. If a video is missing a
  # sub with this extension, a job will be created.
  target_sub_extensions:
    - ".tl.srt"  # Native Target Language (from extraction)
    - ".as.srt"  # Alass Synced
    - ".av.srt"  # Alass Variant AI Synced
    - ".ak.srt"  # KanjiEater/SubPlz Synced
    - ".az.srt"  # AI Generated (Whisper)
    - ".ab.srt"  # Bazarr (base/original downloaded sub)

  # [OPTIONAL] A list of filename parts to ignore during the scan.
  blacklist_filenames: ["OP", "ED", "NCOP"]

  # [OPTIONAL] A list of directory names to completely ignore during the scan.
  blacklist_dirs: ["新しいフォルダー", "New folder", "Specials"]

# ==========================================================
# Settings for the Batch Processor
# ==========================================================
batch_pipeline:
  # Each item in this list is a step that will be run in order
  # for every directory processed by batch.py.
  - name: "Rename 'ja' subs to 'ab' for processing"
    command: ['rename', '-d', '{directory}', '--lang-ext', 'ab', '--lang-ext-original', 'ja', '--unique']

  - name: "Extract & Verify Native Target Language ('ja' -> 'tl')"
    command: ['extract', '-d', '{directory}', '--lang-ext', 'tl', '--lang-ext-original', 'ja', '--verify']

  - name: "Alass Sync ('en' + 'ab' -> 'as')"
    command: ['sync', '-d', '{directory}', '--lang-ext', 'as', '--lang-ext-original', 'en', '--lang-ext-incorrect', 'ab', '--alass']

  - name: "SubPlz Sync ('ab' -> 'ak')"
    command: ['sync', '-d', '{directory}', '--lang-ext', 'ak', '--lang-ext-original', 'ab', '--model', 'large-v3']

  - name: "Generate subs ('az')"
    command: ['gen', '-d', '{directory}', '--lang-ext', 'az', '--model', 'large-v3']

  - name: "Alass Variant AI Sync ('az' + 'ab' -> 'av')"
    command: ['sync', '-d', '{directory}', '--lang-ext', 'av', '--lang-ext-original', 'az', '--lang-ext-incorrect', 'ab', '--alass']

  - name: "Copy best subtitle to 'ja'"
    command: ['copy', '-d', '{directory}', '--lang-ext', 'ja', '--lang-ext-priority', "tl", "as", "av", "ak", "az", "ab", '--overwrite']
```
## How to Use

The system consists of two main commands that can be run in parallel.

### 1. Running the Job Watcher (The Consumer)

This is the main, long-running process you will keep active in the background. It watches your job queue and processes incoming jobs.

*   **To start:** Open a terminal, navigate to your project root, and run:
    ```bash
    subplz watch --config "/path/to/your/config.yml"
    ```
*   Leave this terminal window open. It will now listen for new jobs from any producer and process any jobs that already exist in the queue upon startup.

### 2. Running the Library Scanner (A Producer)

This command is run manually or on a schedule to find media that is missing subtitles and queue it for processing.

*   **To run:** Open a *new* terminal and execute:
    ```bash
    subplz scanner --config "/path/to/your/config.yml"
    ```
*   The script will perform a one-time scan of the library directories defined in your `watcher.path_map`, create job files for any media missing the required subtitles, and then exit. The running `watcher` will see these new files and start processing them immediately.

*   **Scheduling:** For full automation, you can schedule this command to run automatically using:
    *   **Windows:** Task Scheduler
    *   **Linux/macOS:** `cron`

    Example `cron` job to run the scanner every night at 2:00 AM:
    ```crontab
    0 2 * * * /path/to/your/python_env/bin/subplz scanner --config "/path/to/your/config.yml"
    ```

---

## Integration with Bazarr (A Producer)

To make Bazarr a producer, you need to configure its **Custom Post-Processing** feature.

1.  In the Bazarr UI, go to `Settings` -> `Subtitles`.
2.  Scroll down to the **Custom Post-Processing** section.
3.  **Enable** the feature.
4.  In the **Command** box, paste the following one-line command. **Crucially, you must replace `/subplz-jobs/` with the container-side path to your `watcher.jobs` directory.** This path must be a volume mount in your Bazarr Docker container.

    ```bash
    echo "{\"directory\":\"$(echo "{{directory}}" | tr -d '\"')\",\"episode_path\":\"$(echo "{{episode}}" | tr -d '\"')\",\"episode_name\":\"$(echo "{{episode_name}}" | tr -d '\"')\",\"subtitle_path\":\"$(echo "{{subtitles}}" | tr -d '\"')\",\"subtitles_language\":\"$(echo "{{subtitles_language}}" | tr -d '\"')\",\"subtitles_language_code2\":\"$(echo "{{subtitles_language_code2}}" | tr -d '\"')\",\"subtitles_language_code2_dot\":\"$(echo "{{subtitles_language_code2_dot}}" | tr -d '\"')\",\"subtitles_language_code3\":\"$(echo "{{subtitles_language_code3}}" | tr -d '\"')\",\"subtitles_language_code3_dot\":\"$(echo "{{subtitles_language_code3_dot}}" | tr -d '\"')\",\"episode_language\":\"$(echo "{{episode_language}}" | tr -d '\"')\",\"episode_language_code2\":\"$(echo "{{episode_language_code2}}" | tr -d '\"')\",\"episode_language_code3\":\"$(echo "{{episode_language_code3}}" | tr -d '\"')\",\"score\":$(echo "{{score}}" | tr -d '\"'),\"subtitle_id\":\"$(echo "{{subtitle_id}}" | tr -d '\"')\",\"provider\":\"$(echo "{{provider}}" | tr -d '\"')\",\"uploader\":\"$(echo "{{uploader}}" | tr -d '\"')\",\"release_info\":\"$(echo "{{release_info}}" | tr -d '\"')\",\"series_id\":\"$(echo "{{series_id}}" | tr -d '\"')\",\"episode_id\":\"$(echo "{{episode_id}}" | tr -d '\"')\",\"timestamp_utc\":\"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\"}" > /subplz/jobs/"{{episode_name}}".json
    ```
    *Note: We only need the `directory` key for the job file. The filename is appended with `-bazarr.json` for easier identification in the queue.*

5.  **Save** your settings.
6.  If Bazarr is in the docker container or the host, double check that `/subplz/jobs/` exists

Now, whenever Bazarr successfully downloads a new subtitle, it will create a job file in your queue, and your running `watcher` will immediately pick it up for processing.