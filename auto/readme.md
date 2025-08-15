# SubPlz Automation Suite

This suite of scripts provides a robust, queue-based automation system for processing media files with `SubPlz`. It is designed to be triggered by external applications like Bazarr and can also proactively scan your media library to find and fix files that are missing subtitles.

## Core Concepts

This system is built on a **Producer/Consumer** model, which is a powerful and resilient software design pattern.

* **Producer:** An application that creates a "job" that needs to be done. It places a job file into the queue.
    * *Examples:* The **Bazarr Custom Post-Processing Command** and the **`scanner.py`** script are both Producers.
* **Queue:** A shared directory that holds the job files. This decouples the Producers from the Consumers.
    * *Example:* Your `subplz-jobs` folder.
* **Consumer:** A worker process that processes jobs from the queue. In this system, a long-running **`watcher.py`** script acts as a dispatcher, launching a short-lived **`consumer.py`** process for each job.

## Features

* **Real-time Processing:** Instantly process subtitles as soon as they are downloaded by Bazarr.
* **Proactive Library Scanning:** Periodically scan your entire library to find and generate subtitles for files that have none.
* **Resilient Queue System:** If the consumer is offline, jobs from Bazarr will safely accumulate in the queue and will be processed automatically when the consumer restarts. No events are lost.
* **Parallel Processing:** The watcher can launch multiple consumer processes simultaneously to handle a burst of jobs.
* **Highly Configurable:** All paths, commands, and settings are managed in a central `config.yml` file.
* **Clean Architecture:** Each script has a single responsibility, making the system easy to understand and maintain.

## Project Structure

Your automation scripts should be organized in their own directory, for example:

```
automation/
├── config.py         # Centralized configuration loader
├── consumer.py       # The job worker (processes one job)
├── scanner.py        # The library scanner (creates jobs)
└── watcher.py        # The file watcher (launches consumers)
```

## Setup and Installation

1.  **Place the Scripts:** Ensure the four files (`config.py`, `consumer.py`, `scanner.py`, `watcher.py`) are in your `automation/` directory.
2.  **Create `config.yml`:** Create a `config.yml` file in your project's root directory. Copy the contents from the template below and customize it for your system.
3.  **Install Dependencies:** These scripts require `PyYAML` and `watchdog`. Install them using pip:
    ```bash
    pip install PyYAML watchdog
    ```

---

## Configuration (`config.yml`)

This file controls the behavior of all automation scripts.

```yaml
# ===============================================
# Settings for the Real-time Job Consumer/Watcher
# ===============================================
job_consumer_settings:
  # [REQUIRED] The folder to watch for new .json job files.
  job_directory: "D:\\subplz-jobs"

  # [REQUIRED] The command used to run your SubPlz tool.
  subplz_command: "subplz"

  # [REQUIRED] Arguments for the SubPlz command.
  # The "{directory}" placeholder is replaced with the target media directory.
  subplz_args:
    - "helpers/subplz.sh"
    - "{directory}"

  # [REQUIRED] The mapping of paths from inside your Docker containers to your host machine.
  path_map:
    # Key: Path inside the container (must end with a slash)
    # Value: Corresponding path on the host machine
    "/media/": "D:\\Media\\"
    "/data/ja-anime/": "D:\\Media\\J-Anime Shows\\"
    "/subplz-jobs/": "D:\\subplz-jobs\\" # Map the jobs folder too if needed

  # [OPTIONAL] A directory to move job files to if processing fails.
  error_directory: "D:\\subplz-jobs\\failed"


# ==========================================================
# Settings for the Scheduled Library Scanner ("Gap-Filler")
# ==========================================================
scanner_settings:
  # [REQUIRED] A list of the root content directories on your HOST to scan.
  content_dirs:
    - "D:\\Media\\J-Anime Shows\\"
    - "D:\\Media\\J-Shows\\"

  # [REQUIRED] The subtitle extension to check for. If a video is missing a
  # sub with this extension, a job will be created. ".az.srt" is a good
  # choice for AI-generated subtitles.
  target_sub_extension: ".az.srt"

  # [OPTIONAL] A list of filename parts to ignore during the scan.
  # Useful for skipping openings, endings, etc. Case-insensitive.
  blacklist_filenames:
    - "OP"
    - "ED"
    - "NCOP"

  # [OPTIONAL] A list of directory names to completely ignore during the scan.
  blacklist_dirs:
    - "新しいフォルダー"
    - "New folder"
    - "Specials"
```

---

## How to Use

The system consists of two main functions: the long-running watcher and the on-demand scanner.

### 1. Running the Job Watcher

This is the main, long-running process you will keep active in the background. It watches your job queue and dispatches workers.

* **To start:** Open a terminal, navigate to your project root, and run:
    ```bash
    python ./automation/watcher.py config.yml
    ```
* Leave this terminal window open. It will now listen for new jobs from any producer.

### 2. Running the Library Scanner

This script is run manually or on a schedule to find media that Bazarr missed and queue it for processing.

* **To run:** Open a *new* terminal and execute:
    ```bash
    python ./automation/scanner.py config.yml
    ```
* The script will perform a one-time scan of your library, create job files for any media missing subtitles, and then exit. The running `watcher` will see these new files and start processing them.

* **Scheduling:** For full automation, you can schedule this command to run automatically using:
    * **Windows:** Task Scheduler
    * **Linux/macOS:** `cron`

---

## Integration with Bazarr (Producer Setup)

To make Bazarr a producer, you need to configure its **Custom Post-Processing** feature.

1.  In the Bazarr UI, go to `Settings` -> `Subtitles`.
2.  Scroll down to the **Custom Post-Processing** section.
3.  **Enable** the feature.
4.  In the **Command** box, paste the following one-line command. Make sure to replace `/subplz-jobs/` if you used a different mount point in your `docker-compose.yml`.

    ```bash
    echo "{\"directory\":\"$(echo "{{directory}}" | tr -d '\"')\",\"episode_path\":\"$(echo "{{episode}}" | tr -d '\"')\",\"episode_name\":\"$(echo "{{episode_name}}" | tr -d '\"')\",\"subtitle_path\":\"$(echo "{{subtitles}}" | tr -d '\"')\",\"subtitles_language\":\"$(echo "{{subtitles_language}}" | tr -d '\"')\",\"subtitles_language_code2\":\"$(echo "{{subtitles_language_code2}}" | tr -d '\"')\",\"subtitles_language_code2_dot\":\"$(echo "{{subtitles_language_code2_dot}}" | tr -d '\"')\",\"subtitles_language_code3\":\"$(echo "{{subtitles_language_code3}}" | tr -d '\"')\",\"subtitles_language_code3_dot\":\"$(echo "{{subtitles_language_code3_dot}}" | tr -d '\"')\",\"episode_language\":\"$(echo "{{episode_language}}" | tr -d '\"')\",\"episode_language_code2\":\"$(echo "{{episode_language_code2}}" | tr -d '\"')\",\"episode_language_code3\":\"$(echo "{{episode_language_code3}}" | tr -d '\"')\",\"score\":$(echo "{{score}}" | tr -d '\"'),\"subtitle_id\":\"$(echo "{{subtitle_id}}" | tr -d '\"')\",\"provider\":\"$(echo "{{provider}}" | tr -d '\"')\",\"uploader\":\"$(echo "{{uploader}}" | tr -d '\"')\",\"release_info\":\"$(echo "{{release_info}}" | tr -d '\"')\",\"series_id\":\"$(echo "{{series_id}}" | tr -d '\"')\",\"episode_id\":\"$(echo "{{episode_id}}" | tr -d '\"')\",\"timestamp_utc\":\"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\"}" > /subplz-jobs/"{{episode_name}}".json
    ```
5.  **Save** your settings.

Now, whenever Bazarr successfully downloads a new subtitle, it will create a job file, and your running `watcher` will immediately pick it up for processing by `SubPlz`.