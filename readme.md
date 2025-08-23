# SubPlz🫴: Get Incredibly Accurate Subs for Anything

https://user-images.githubusercontent.com/32607317/219973521-5a5c2bf2-4df1-422b-874c-5731b4261736.mp4

🫴 Generate, sync, and manage subtitle files for any media; Generate your own audiobook subs similar to Kindle's Immersion Reading 📖🎧

## Table of Contents

- [SubPlz🫴: Get Incredibly Accurate Subs for Anything](#subplz-get-incredibly-accurate-subs-for-anything)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
- [Installation](#installation)
  - [Running with Docker (Recommended)](#running-with-docker-recommended)
    - [Prerequisites](#prerequisites)
      - [1. Update Docker Desktop (Windows WSL2 Users)](#1-update-docker-desktop-windows-wsl2-users)
      - [Check NVIDIA Driver and CUDA Version](#check-nvidia-driver-and-cuda-version)
    - [Docker Compose Setup](#docker-compose-setup)
    - [Building and Running](#building-and-running)
    - [Test Your SubPlz GPU Setup](#test-your-subplz-gpu-setup)
    - [Troubleshooting Docker GPU Issues](#troubleshooting-docker-gpu-issues)
  - [Setup from source](#setup-from-source)
  - [Run from Colab](#run-from-colab)
- [How to Use](#how-to-use)
  - [Quick Guide](#quick-guide)
    - [Sync](#sync)
    - [Gen](#gen)
    - [Batch](#batch)
    - [Alass](#alass)
    - [Rename (When Sub Names Don't Match)](#rename-when-sub-names-dont-match)
    - [Rename a Language Extension (When Sub Names Match)](#rename-a-language-extension-when-sub-names-match)
  - [Usage Notes](#usage-notes)
  - [Sort Order](#sort-order)
  - [Overwrite](#overwrite)
  - [Tuning Recommendations](#tuning-recommendations)
    - [For Audiobooks](#for-audiobooks)
    - [For Realigning Subtitles](#for-realigning-subtitles)
- [Automation Suite](#automation-suite)
  - [Quick Start](#quick-start)
  - [Core Concepts](#core-concepts)
  - [Features](#features-1)
  - [Configuration (`config.yml`)](#configuration-configyml)
  - [Running the Automation Commands](#running-the-automation-commands)
    - [1. Running the Job Watcher (The Consumer)](#1-running-the-job-watcher-the-consumer)
    - [2. Running the Library Scanner (A Producer)](#2-running-the-library-scanner-a-producer)
  - [Integration with Bazarr (A Producer)](#integration-with-bazarr-a-producer)
- [Generating All Subtitle Algorithms in Batch](#generating-all-subtitle-algorithms-in-batch)
- [Anki Support](#anki-support)
  - [Setup Instructions](#setup-instructions)
- [FAQ](#faq)
  - [Can I run this with multiple Audio files and *One* script?](#can-i-run-this-with-multiple-audio-files-and-one-script)
  - [How do I get a bunch of MP3's into one file then?](#how-do-i-get-a-bunch-of-mp3s-into-one-file-then)
- [Technologies \& Techniques](#technologies--techniques)
- [Support](#support)
- [Thanks](#thanks)
- [Other Cool Projects](#other-cool-projects)

## Features

- **Sync Existing Subtitles**: Multiple options to automate synchronizing your subtitles with various techniques in bulk
- **Powerful Automation Suite**: Includes a real-time file watcher (`watch`) and a scheduled library scanner (`scanner`) to fully automate your subtitle processing pipeline.
- **Accurately Subtitle Narrated Media**: Leverage the original source text of an ebook to provide highly accurate subtitles
- **Create New Subtitles**: Generate new subtitles from the audio of a media file
- **File Management**: Automatically organize and rename your subtitles to match your media files
- **Provide Multiple Video Subtitle Options**: Combines other features to allow you to have multiple alignment & generation subs available to you, for easy auto-selecting of your preferred version (dependent on video player support)

# Installation

Currently supports Docker (preferred), Windows, and unix based OS's like Ubuntu 24.04 (even WSL2). Primarily supports Japanese, but other languages may work as well with limited dev support.

## Running with Docker (Recommended)

Docker with GPU support can be challenging to set up correctly. This guide will help you verify your setup works before running SubPlz. The docker image *does* work with GPU, so if you have issues GPT, Gemini, and Claude are your friends.

### Prerequisites

Before setting up Docker, verify your host system has the required components:

#### 1. Update Docker Desktop (Windows WSL2 Users)

**CRITICAL for Windows with WSL2:** Update Docker Desktop to version **4.44.2 or later**. Older versions have CUDA symbol resolution issues that prevent GPU access.

- **Known working:** Docker Desktop 4.44.2+
- **Known broken:** Docker Desktop 4.24.2 and earlier versions

#### Check NVIDIA Driver and CUDA Version

```bash
# Check your NVIDIA driver version and CUDA support
nvidia-smi

# Should show something like:
# Driver Version: 576.88      CUDA Version: 12.9
```

### Docker Compose Setup

1. Create a `compose.yml` file in your project directory:

```yaml
services:
  subplz:
    image: kanjieater/subplz:latest
    container_name: subplz
    restart: unless-stopped
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/Chicago
      - WHISPER_MODEL=turbo
      - HF_HOME=/config/cache/huggingface
    volumes:
      # Map your media library. Left side is your Host path, right is the Container path.
      - "/path/on/your/host/media:/media"
      # Map a local config directory to the container.
      - "./config:/config"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

2. It is also recommended to immediately put this in your `./config/` folder: [Configuration (`config.yml`)](#configuration-configyml)


### Building and Running

3. **Start the service:**

   ```bash
   docker compose up -d
   ```

4. **View logs:**

   ```bash
   docker compose logs -f subplz
   ```

### Test Your SubPlz GPU Setup

Before running the full application, test that your built SubPlz image can access the GPU:

```bash
# Test GPU access with your SubPlz image
docker run -it --rm --gpus all --entrypoint python subplz -c "import torch; print(torch.__version__); print(f'CUDA available: {torch.cuda.is_available()}');"
```

**Expected output:**

```
2.8.0+cu128
CUDA available: True
```

If this test fails, **do not proceed** until GPU access is working.

### Troubleshooting Docker GPU Issues

**Problem: "CUDA initialization: Unexpected error"**

- **Cause:** CUDA version mismatch between host driver and container runtime
- **Solution:** Update the Dockerfile to use a PyTorch image matching your CUDA version:
  ```dockerfile
  # Check your host CUDA version with: nvidia-smi
  # Then use matching PyTorch image, e.g.:
  FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime  # For CUDA 12.9
  ```

**Problem: "docker: Error response from daemon: could not select device driver"**

- **Cause:** NVIDIA Container Toolkit not installed
- **Solution:** Install nvidia-docker2:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install nvidia-docker2
  sudo systemctl restart docker
  ```

**Problem: "CUDA available: False" in test command**

- **Windows WSL2:** Update Docker Desktop to 4.44.2 or later
- **Linux:** Ensure nvidia-container-toolkit is installed and Docker daemon restarted
- **All platforms:** Verify `nvidia-smi` works on the host first

**Problem: Permission denied on mounted volumes**

- **Cause:** User ID mismatch between container and host
- **Solution:** Update the `user` field in docker-compose.yml:
  ```yaml
  user: "${UID:-1000}:${GID:-1000}"  # Uses your actual user ID
  ```

**Problem: Symlink errors during model download**

- **Cause:** Filesystem doesn't support symlinks (common with network mounts)
- **Solution:** Uncomment the symlink environment variables in docker-compose.yml:
  ```yaml
  environment:
    - HF_HUB_DISABLE_SYMLINKS=1
    - HF_HUB_DISABLE_SYMLINKS_WARNING=1
  ```

## Setup from source

1. Install `ffmpeg` and make it available on the path

2. `git clone https://github.com/kanjieater/SubPlz.git`

3. Use python >= `3.11.2` (latest working version is always specified in `pyproject.toml`)

4. `pip install .`

5. You can get a full list of cli params from `subplz sync -h`

6. If you're using a single file for the entire audiobook with chapters you are good to go. If an file with audio is too long it may use up all of your RAM. You can use the docker image [`m4b-tool`](https://github.com/sandreas/m4b-tool#installation) to make a chaptered audio file. Trust me, you want the improved codec's that are included in the docker image. I tested both and noticed a huge drop in sound quality without them. When lossy formats like mp3 are transcoded they lose quality so it's important to use the docker image to retain the best quality if you plan to listen to the audio file.

## Run from Colab

1. Open this [Colab](https://colab.research.google.com/drive/1LOu6tffvYiOqzrSMH6Pe91Eka55uOuT3?usp=sharing)
2. In Google Drive, create a folder named `sync` on the root of MyDrive
3. Upload the audio/video file and supported text to your `sync` folder
4. Open the colab, you can change the last line if you want, like `-d "/content/drive/MyDrive/sync/Harry Potter 1/"` for the quick guide example
5. In the upper menu, click Runtime > run all, give the necessary permissions and wait for it to finish, should take some 30 min for your average book

# How to Use

## Quick Guide

### Sync

1. Put an audio/video file and a text file in a folder.
   1. Audio / Video files: `m4b`, `mkv` or any other audio/video file
   2. Text files: `srt`, `vtt`, `ass`, `txt`, or `epub`

```bash
/sync/
└── /Harry Potter 1/
   ├── Im an audio file.m4b
   └── Harry Potter.epub
└── /Harry Potter 2 The Spooky Sequel/
   ├── Harry Potter 2 The Spooky Sequel.mp3
   └── script.txt
```

2. List the directories you want to run this on. The `-d` parameter can multiple audiobooks to process like: `subplz sync -d "/mnt/d/sync/Harry Potter 1/" "/mnt/d/sync/Harry Potter 2 The Spooky Sequel/"`
3. Run `subplz sync -d "<full folder path>"` using something like `/mnt/d/sync/Harry Potter 1`
4. From there, use a [texthooker](https://github.com/Renji-XD/texthooker-ui) with something like [mpv_websocket](https://github.com/kuroahna/mpv_websocket) and enjoy Immersion Reading.

### Gen

1. Put an audio/video file and a text file in a folder.
   1. Audio / Video files: `m4b`, `mkv` or any other audio/video file

```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising EP00.mkv
   └── NeoOtaku Uprising EP01.avi
```

1. List the directories you want to run this on. The `-d` parameter can multiple files to process like: `subplz gen -d "/mnt/d/NeoOtaku Uprising The Anime" --model turbo`
2. Run `subplz gen -d "<full folder path>" --model turbo` using something like `/mnt/d/sync/NeoOtaku Uprising The Anime`. Large models are highly recommended for `gen` (unlike `sync`)
3. From there, use a video player like MPV or Plex. You can also use `--lang-ext az` to set a language you wouldn't otherwise need as a designated "AI subtitle", and use it as a fallback when sync doesn't work or you don't have existing subtitles already

### Batch

1. Define your desired workflow in the `batch_pipeline` section of your `config.yml`. See the [Automation Suite](#automation-suite) section for configuration examples.
2. Place media files that you want to process in a directory.

```bash
/media/
└── /My Awesome Anime/
   ├── Episode 01.mkv
   └── Episode 01.ja.srt
```

3. Run `subplz batch -d "<full folder path>" -c "/path/to/your/config.yml"`. This will execute every step in your batch_pipeline against the target directory.
4. The output will be a combination of all the steps in your pipeline. For example, if your pipeline includes `extract`, `sync`, `alass`, and `gen`, the final directory might look like this:

```bash
/media/
└── /My Awesome Anime/
   ├── Episode 01.mkv
   ├── Episode 01.ja.srt (Original subtitle file)
   ├── Episode 01.en.srt (Result of extract - embedded subs)
   ├── Episode 01.ak.srt (Result of sync - SubPlz AI alignment)
   ├── Episode 01.as.srt (Result of alass - Alass alignment)
   └── Episode 01.az.srt (Result of gen - FasterWhisper generation)
```

This gives you multiple subtitle options to choose from, allowing you to quickly switch between different algorithms while watching to find the best one for your content.

### Alass

1. Put a video(s) with embdedded subs & sub file(s) that need alignment in a folder.
   1. Video & Sub files: `m4b`, `mkv` or any other audio/video file
   2. If you don't have embdedded subs, you'll need it to have a `*.en.srt` extension in the folder
   3. Consider using Rename to get your files ready for alass

```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.srt
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.en.srt
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.srt
```

1. List the directories you want to run this on. The `-d` parameter can multiple files to process like: `subplz sync -d "/mnt/d/NeoOtaku Uprising The Anime" --alass --lang-ext "ja" --lang-ext-original "en"`
   1. You could also add `--lang-ext-incorrect "ja"` if you had `NeoOtaku Uprising With No Embedded Eng Subs EP01.ja.srt` instead of `NeoOtaku Uprising With No Embedded Eng Subs EP01.srt`. This is the incorrect timed sub from Alass
2. From there, SubPlz will extract the first available subs from videos writing them with `--lang-ext-original` extension, make sure the subtitles are sanitized, convert subs to the same format for Alass if need be, and align the incorrect timings with the timed subs to give you correctly timed subs like below:

```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.en.srt (embedded)
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.ja.srt (timed)
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.srt (original/incorrect timings)
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.en.srt (no change)
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.ja.srt (timed)
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.srt (original/incorrect timings)
```

### Rename (When Sub Names Don't Match)

1. Put a video(s) & sub file(s) that need alignment in a folder.
   1. Video & Sub files: `m4b`, `mkv` or any other audio/video file

```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── 1.srt
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   └── 2.ass
```

1. Run `subplz rename -d "/mnt/v/Videos/J-Anime Shows/NeoOtaku Uprising The Anime/" --lang-ext "ab" --dry-run` to see what the rename would be
2. If the renames look right, run it again without the `--dry-run` flag: `subplz rename -d "/mnt/v/Videos/J-Anime Shows/NeoOtaku Uprising The Anime/" --lang-ext ab`

```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.ab.srt
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.ab.ass
```

### Rename a Language Extension (When Sub Names Match)

1. Put a video(s) & sub file(s) that match names in a folder.
   1. Video & Sub files: `m4b`, `mkv` or any other audio/video file
   2. The names must be exactly the same besides language extension & hearing impaired code

```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.ab.cc.srt (notice the hearing impaired cc)
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.ab.srt
```

1. Run `subplz rename -d "/mnt/v/Videos/J-Anime Shows/NeoOtaku Uprising The Anime/" --lang-ext jp --lang-ext-original ab` to get:

```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.jp.srt (notice the removed cc)
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.jp.srt
```

## Usage Notes

- This can be GPU intense, RAM intense, and CPU intense script part. `subplz sync -d "<full folder path>"` eg `subplz sync -d "/mnt/d/Editing/Audiobooks/かがみの孤城/"`. This runs each file to get a character level transcript. It then creates a sub format that can be matched to the `script.txt`. Each character level subtitle is merged into a phrase level, and your result should be a `<name>.srt` file. The video or audio file then can be watched with `MPV`, playing audio in time with the subtitle.
- CUDA-enabled GPU is highly recommended for the best performance, especially with larger and more accurate models.

## Sort Order

By default, the `-d` parameter will pick up the supported files in the directory(s) given. Ensure that your OS sorts them in an order that you would want them to be patched together in. Sort them by name, and as long as all of the audio files are in order and the all of the text files are in the same order, they'll be "zipped" up individually with each other.

## Overwrite

By default the tool will overwrite any existing srt named after the audio file's name. If you don't want it to do this you must explicitly tell it not to.

```bash
subplz sync -d "/mnt/v/somefolder" --no-overwrite
```

## Tuning Recommendations

For different use cases, different parameters may be optimal.

### For Audiobooks

- **Recommended**: `subplz sync -d "/mnt/d/sync/Harry Potter"`
- A chapter `m4b` file will allow us to split up the audio and do things in parallel
- There can be slight variations between `epub` and `txt` files, like where full character spaces aren't pickedup in `epub` but are in `txt`. A chaptered `epub` may be faster, but you can have more control over what text gets synced from a `txt` file if you need to manually remove things (but `epub` is still probably the easier option, and very reliable)
- If the audio and the text differ greatly - like full sections of the book are read in different order, you will want to use `--no-respect-grouping` to let the algorithm remove content for you
- The default `--model "tiny"` seems to work well, and is much faster than other models. If your transcript is inaccurate, consider using a larger model to compensate

### For Realigning Subtitles

- **Recommended**: `subplz sync --model turbo -d "/mnt/v/Videos/J-Anime Shows/Sousou no Frieren"`
- Highly recommend running with something like `--model "turbo"` as subtitles often have sound effects or other things that won't be picked up by transcription models. By using a large model, it will take much longer (a 24 min episode can go from 30 seconds to 4 mins for me), but it will be much more accurate.
- Subs can be cut off in strange ways if you have an unreliable transcript, so you may want to use `--respect-grouping`. If you find your subs frequently have very long subtitle lines, consider using `--no-respect-grouping`

# Automation Suite

This suite provides a robust, queue-based automation system for processing media files with `SubPlz`. It is a fully integrated part of the `subplz` command-line tool, designed to be triggered by external applications like Bazarr and to proactively scan your media library for missing subtitles.

## Quick Start

1. **Create a Configuration Directory:**
   ```bash
   mkdir -p /path/to/your/subplz_config
   ```
2. **Create `config.yml`:** Inside that new directory, create a `config.yml` file and paste the full configuration template from the section below.
3. **Customize `config.yml`:** Edit the `base_dirs` and `path_map` to match the real paths on your host machine.
4. **Set Environment Variable:** Set the `BASE_PATH` environment variable so the application can find your config directory.
   ```bash
   export BASE_PATH=/path/to/your/subplz_config
   ```
5. **Run the Watcher:**
   ```bash
   subplz watch
   ```

## Core Concepts

This system is built on a **Producer/Consumer** model, a powerful and resilient software design pattern.

* **Producer:** An application that creates a "job" that needs to be done. It places a job file into a designated queue folder.
  * *Examples:* The **Bazarr Custom Post-Processing** feature and the **`subplz scanner`** command are both Producers.
* **Queue:** A shared directory that holds job files (in `.json` format). This decouples the Producers from the Consumer, allowing jobs to accumulate safely even if the consumer is offline.
* **Consumer:** The long-running **`subplz watch`** command. It continuously monitors the queue, picking up one job at a time and processing it through a user-defined pipeline.

## Features

* **Real-time Processing:** Instantly process media files as soon as a job is created by a producer like Bazarr.
* **Proactive Library Scanning:** Use the `scanner` command on a schedule (e.g., cron job) to find and queue jobs for media files that are missing subtitles.
* **Powerful Processing Pipelines:** Define complex, multi-step processing workflows in your `config.yml` that are executed for every job.
* **Resilient Queue System:** If the watcher is offline, jobs will safely accumulate in the queue and will be processed automatically when it restarts. No events are lost.
* **Process-Safe Logging:** All components, including the watcher and scanner, log to the same rotating log file, providing a complete, interleaved history of all system activity.
* **Highly Configurable:** All paths, settings, and the entire processing pipeline are managed in a central `config.yml` file.
* **Integrated Tooling:** The watcher and scanner are not separate scripts but are first-class commands within the `subplz` CLI.

-----

## Configuration (`config.yml`)

**IMPORTANT:** All paths in the `base_dirs` section are relative to a root directory that you define with the `BASE_PATH` environment variable. This allows you to keep all your configuration, logs, and job files together in one portable folder.

This single file controls the behavior of all automation commands and the processing pipeline.

```yaml
# Rooted from env var BASE_PATH
base_dirs:
  logs: "logs"
  cache: "cache"
  # [REQUIRED] The folder on the HOST machine to watch for new .json job files.
  watcher_jobs: "jobs"
  # [REQUIRED] A directory on the HOST to move job files to if processing fails.
  watcher_errors: "fails"

# ===============================================
# Settings for the Real-time Job Consumer/Watcher
# ===============================================
watcher:
  # [REQUIRED] The mapping of paths from inside your Docker containers to your host machine.
  # This allows the script to translate container paths from Bazarr jobs to real host paths.
  path_map:
    # Key: Path inside the container (must end with a slash)
    # Value: Corresponding path on the host machine
    "/data/ja-anime/": "/mnt/an/ja-anime/"
    # "/scan_only/unmapped_media/": "/home/ke/unmapped_media/"


  # [OPTIONAL] If this key is present, the watcher will use a polling-based
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
    - ".ae.srt"  # Native Target Language (from extraction)
    - ".as.srt"  # Alass Synced
    - ".av.srt"  # Alass Variant AI Synced
    - ".ak.srt"  # KanjiEater/SubPlz Synced
    - ".az.srt"  # AI Generated (Whisper)
    - ".ab.srt"  # Bazarr (base/original downloaded sub)

  # [OPTIONAL] A list of filename parts to ignore during the scan.
  blacklist_filenames: ["OP", "ED", "NCOP", "NCED"]

  # [OPTIONAL] A list of directory names to completely ignore during the scan.
  blacklist_dirs: ["新しいフォルダー", "New folder"]

# ==========================================================
# Settings for the Batch Processor
# ==========================================================
batch_pipeline:
  - name: "Source: Rename 'ja' subs to 'ab' for processing"
    command: 'rename -d "{directory}" --file "{file}" --lang-ext ab --lang-ext-original ja --unique --overwrite'

  - name: "Embedded: Extract & Verify Native Target Language ('ja' -> 'ae')"
    command: 'extract -d "{directory}" --file "{file}" --lang-ext ae --lang-ext-original ja --verify'

  - name: "Alass: ('en' + 'ab' -> 'as')"
    command: 'sync -d "{directory}" --file "{file}" --lang-ext as --lang-ext-original en --lang-ext-incorrect ab --alass'

  - name: "SubPlz: ('ab' -> 'ak')"
    command: 'sync -d "{directory}" --file "{file}" --lang-ext ak --lang-ext-original ab --model turbo'

  - name: "Stable-ts: ('az')"
    command: 'gen -d "{directory}" --file "{file}" --lang-ext az --model turbo'

  - name: "Alass Variant: ('az' + 'ab' -> 'av')"
    command: 'sync -d "{directory}" --file "{file}" --lang-ext av --lang-ext-original az --lang-ext-incorrect ab --alass'

  - name: "Best: Copy best subtitle to 'ja'"
    command: 'copy -d "{directory}" --file "{file}" --lang-ext ja --lang-ext-priority ae av as ak az ab --overwrite'
```

## Running the Automation Commands

The system consists of two main commands that can be run in parallel.

### 1. Running the Job Watcher (The Consumer)

This is the main, long-running process you will keep active in the background. It watches your job queue and processes incoming jobs.

* **Docker:**
  ```bash
  docker compose run --rm subplz watch --config /config/config.yml
  ```
* **Local:**
  ```bash
  subplz watch --config "/path/to/your/config.yml"
  ```
* Leave this terminal window open. It will now listen for new jobs from any producer and process any jobs that already exist in the queue upon startup.

### 2. Running the Library Scanner (A Producer)

This command is run manually or on a schedule to find media that is missing subtitles and queue it for processing.

* **Docker:**

  ```bash
  docker compose run --rm subplz scanner --config /config/config.yml
  ```

* **Local:**

  ```bash
  subplz scanner --config "/path/to/your/config.yml"
  ```

* The script will perform a one-time scan of the library directories defined in your `watcher.path_map`, create job files for any media missing the required subtitles, and then exit. The running `watcher` will see these new files and start processing them immediately.

* **Scheduling:** For full automation, you can schedule this command to run automatically using:

  * **Windows:** Task Scheduler
  * **Linux/macOS:** `cron`

  Example `cron` job to run the scanner every night at 2:00 AM:

  ```crontab
  0 2 * * * /path/to/your/python_env/bin/subplz scanner --config "/path/to/your/config.yml"
  ```

-----

## Integration with Bazarr (A Producer)

To make Bazarr a producer, you need to configure its **Custom Post-Processing** feature.

1. In the Bazarr UI, go to `Settings` -> `Subtitles`.

2. Scroll down to the **Custom Post-Processing** section.

3. **Enable** the feature.

4. In the **Command** box, paste the following one-line command. **Crucially, you must replace `/subplz/jobs/` with the container-side path to your `watcher.jobs` directory.** This path must be a volume mount in your Bazarr Docker container.

   ```bash
   echo "{\"directory\":\"$(echo "{{directory}}" | tr -d '\"')\",\"episode_path\":\"$(echo "{{episode}}" | tr -d '\"')\",\"episode_name\":\"$(echo "{{episode_name}}" | tr -d '\"')\",\"subtitle_path\":\"$(echo "{{subtitles}}" | tr -d '\"')\",\"subtitles_language\":\"$(echo "{{subtitles_language}}" | tr -d '\"')\",\"subtitles_language_code2\":\"$(echo "{{subtitles_language_code2}}" | tr -d '\"')\",\"subtitles_language_code2_dot\":\"$(echo "{{subtitles_language_code2_dot}}" | tr -d '\"')\",\"subtitles_language_code3\":\"$(echo "{{subtitles_language_code3}}" | tr -d '\"')\",\"subtitles_language_code3_dot\":\"$(echo "{{subtitles_language_code3_dot}}" | tr -d '\"')\",\"episode_language\":\"$(echo "{{episode_language}}" | tr -d '\"')\",\"episode_language_code2\":\"$(echo "{{episode_language_code2}}" | tr -d '\"')\",\"episode_language_code3\":\"$(echo "{{episode_language_code3}}" | tr -d '\"')\",\"score\":$(echo "{{score}}" | tr -d '\"'),\"subtitle_id\":\"$(echo "{{subtitle_id}}" | tr -d '\"')\",\"provider\":\"$(echo "{{provider}}" | tr -d '\"')\",\"uploader\":\"$(echo "{{uploader}}" | tr -d '\"')\",\"release_info\":\"$(echo "{{release_info}}" | tr -d '\"')\",\"series_id\":\"$(echo "{{series_id}}" | tr -d '\"')\",\"episode_id\":\"$(echo "{{episode_id}}" | tr -d '\"')\",\"timestamp_utc\":\"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\"}" > /subplz/jobs/"{{episode_name}}".json
   ```

   *Note: We only need the `directory` key for the job file. The filename is appended with `-bazarr.json` for easier identification in the queue.*

5. **Save** your settings.

6. If Bazarr is in the docker container or the host, double check that `/subplz/jobs/` exists

Now, whenever Bazarr successfully downloads a new subtitle, it will create a job file in your queue, and your running `watcher` will immediately pick it up for processing.

# Generating All Subtitle Algorithms in Batch

Let's say you want to automate getting the best subs for every piece of media in your library. SubPlz takes advantage of how well video players integrate with language codes by overriding them to map them to algorithms, instead of different languages. This makes it so you can quickly switch between a sub on the fly while watching content, and easily update your preferred option for a series later on if your default doesn't work.

Just run `subplz batch --config /path/to/config.yml`, as it's documented in the readme, with a sub like `sub1.ja.srt` and `video1.mkv` and it will genearate the following:

| Algorithm | Default Language Code | Mnemonic | Description |
| -------- | ------- | -------- | ------- |
| Bazarr | ab | B for Bazarr | Default potentially untimed subs in target language|
| Alass | as | S for Ala_ss_ | Subs that have been aligned using `en` & `ab` subs via Alass |
| SubPlz | ak | K for KanjiEater | Generated alignment from AI with the `ab` subs text |
| Alass Variant | av | V for variant | Ignore the embedded subs, and use subs that have been aligned using `az` for timing and `ab` subs content via Alass|
| FasterWhisper | az | Z for the last option | Generated purely based on audio. Surprisingly accurate but not perfect. |
| Original | en | Animes subs tend to be in EN | This would be the original timings used for Alass, and what would be extracted from you videos automatically|
| Preferred | ja | Your target language | This is a copy of one of the other options, named with your target language so it plays this by default |
| Embedded | ae | Target Language (that was embedded) | Explicitly extract the target language sub from your media if it existed so you can use it as the Preferred sub in a media player like Plex by default |

# Anki Support

- Generates subs2srs style deck
- Imports the deck into Anki automatically

The Anki support currently takes your m4b file in `<full_folder_path>` named `<name>.m4b`, where `<name>` is the name of the media, and it outputs srs audio and a TSV file that can is sent via AnkiConnect to Anki. This is useful for searching across [GoldenDict](https://www.youtube.com/playlist?list=PLV9y64Yrq5i-1ztReLQQ2oyg43uoeyri-) to find sentences that use a word, or to merge automatically with custom scripts (more releases to support this coming hopefully).

## Setup Instructions

1. Install ankiconnect add-on to Anki.
2. I recommend using `ANKICONNECT` as an environment variable. Set `export ANKICONNECT=localhost:8755` or `export ANKICONNECT="$(hostname).local:8765"` in your `~/.zshrc` or bashrc & activate it.
3. Just like the line above, Set `ANKI_MEDIA_DIR` to your anki profile's media path: `export ANKI_MEDIA_DIR="/mnt/f/Anki2/KanjiEater/collection.media/"`. You need to change this path.
4. Make sure you are in the project directory `cd ./AudiobookTextSync`
5. Install the main project `pip install .` (only needs to be done once)
6. Install `pip install .[anki]` (only needs to be done once)
7. Copy the file from the project in `./anki_importer/mapping.template.json` to `./anki_importer/mapping.json`. `mapping.json` is your personal configuration file that you can and should modify to set the mapping of fields that you want populated.
   My actual config looks like this:

```json
{
  "deckName": "!優先::Y メディア::本",
  "modelName": "JapaneseNote",
  "fields": {
    "Audio": 3,
    "Expression": 1,
    "Vocab": ""
  },
  "options": {
    "allowDuplicate": true
  },
  "tags": [
    "mmi",
    "suspendMe"
  ]
}
```

The number next to the Expression and Audio maps to the fields like so:

```
1: Text of subtitle: `パルスに援軍を求めてきたのである。`
2: Timestamps of sub: `90492-92868`
3: Sound file: `[sound:アルスラーン戦記9　旌旗流転_90492-92868.mp3]`
4: Image (not very really useful for audiobooks): <img src='アルスラーン戦記9　旌旗流転_90492-92868.jpg'>
5: Sub file name: アルスラーン戦記9　旌旗流転.m4b,アルスラーン戦記9　旌旗流転.srt
```

Notice you can also set fields and tags manually. You can set multiple tags. Or like in my example, you can set `Vocab` to be empty, even though it's my first field in Anki.
8. Run the command below

**IMPORTANT** Currently the folder, `m4b`, and `srt` file must share the same name. So:

```
/sync/
└── /NeoOtaku Uprising Audiobook/
   ├── NeoOtaku Uprising Audiobook.m4b
   ├── NeoOtaku Uprising Audiobook.srt
```

**Command:**

```bash
./anki_importer/anki.sh "<full_folder_path>"
```

**Example:**

```bash
./anki_importer/anki.sh "/mnt/d/sync/kokoro/"
```

# FAQ

## Can I run this with multiple Audio files and *One* script?

It's not recommended. You will have a bad time.

If your audiobook is huge (eg 38 hours long & 31 audio files), then break up each section into an m4b or audio file with a text file for it: one text file per one audio file. This will work fine.

But it *can* work in very specific circumstances. The exception to the Sort Order rule, is if we find one transcript and multiple audio files. We'll assume that's something like a bunch of `mp3`s or other audio files that you want to sync to a single transcript like an `epub`. This only works if the `epub` chapters and the `mp3` match. `Txt` files don't work very well for this case currently. I still don't recommend it.

## How do I get a bunch of MP3's into one file then?

Please use m4b for audiobooks. I know you may have gotten them in mp3 and it's an extra step, but it's *the* audiobook format.

I've heard of people using https://github.com/yermak/AudioBookConverter

Personally, I use the docker image for [`m4b-tool`](https://github.com/sandreas/m4b-tool#installation). If you go down this route, make sure you use the docker version of m4b-tool as the improved codecs are included in it. I tested m4b-tool without the docker image and noticed a huge drop in sound quality without them. When lossy formats like mp3 are transcoded they lose quality so it's important to use the docker image to retain the best quality. I use the `helpers/merge2.sh` to merge audiobooks together in batch with this method.

Alternatively you could use ChatGPT to help you combine them. Something like this:

```bash
!for f in "/content/drive/MyDrive/name/成瀬は天下を取りに行く/"*.mp3; do echo "file '$f'" >> mylist.txt; done
!ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp3
```

# Technologies & Techniques

For a glimpse of some of the technologies & techniques we're using depending on the arguments, here's a short list:

- **faster-whisper**: for using AI to generate subtitles fast
- **stable-ts**: for more accurate Whisper time stamps
- **Silero VAD**: for Voice Activity Detection
- **Alass**: for language agnostic subtitle alignment
- **Needleman–Wunsch algorithm**: for alignment to original source text

Currently I am only developing this tool for Japanese use, though rumor has it, the `lang` flag can be used for other languages too.

It requires a modern GPU with decent VRAM, CPU, and RAM. There's also a community built Google Colab notebook available on discord.

Current State of SubPlz alignments:

- The subtitle timings will be 98% accurate for most intended use cases
- The timings will be mostly accurate, but may come late or leave early
- Occasionally, non-spoken things like character names at the start of a line or sound effects in subtitles will be combined with other lines
- Theme songs might throw subs off time, but will auto-recover almost immediately after
- Known Issues: RAM usage. 5+ hr audiobooks can take more than 12 GB of RAM. I can't run a 19 hr one with 48GB of RAM. The current work around is to use an epub + chaptered m4b audiobook. Then we can automatically split the ebook text and audiobook chapters to sync in smaller chunks accurately. Alternatively you could use multiple text files and mp3 files to achieve a similar result.

How does this compare to Alass for video subtitles?

- Alass is usually either 100% right once it get's lined up - or way off and unusable. In contrast, SubPlz is probably right 98% but may have a few of the above issues. Ideally you'd have both types of subtitle available and could switch from an Alass version to a SubPlz version if need be. Alternatively, since SubPlz is consistent, you could just default to always using it, if you find it to be "good enough". [See Generating All Subtitle Algorithms in Batch](#generating-all-subtitle-algorithms-in-batch)

Current State of Alass alignments:

- Alass tends to struggle on large commercial gaps often found in Japanese TV subs like AT-X
- Once Alass get's thrown off it may stay misaligned for the rest of the episode
- SubPlz can extract the first subtitle embedded, but doesn't try to get the longest one. Sometimes you'll get Informational or Commentary subs which can't be used for alignments of the spoken dialog. We may be able to automate this in the future

# Support

Support for this tool can be found [on KanjiEater's thread](https://discord.com/channels/617136488840429598/1076966013268148314) or community support on [The Moe Way Discord](https://learnjapanese.moe/join/)

Support for any tool by KanjiEater can be found [on KanjiEater's Discord](https://discord.com/invite/agbwB4p)

The Deep Weeb Podcast - Sub Please 😉

<a href="https://youtube.com/c/kanjieater"><img src="https://github.com/gauravghongde/social-icons/blob/master/SVG/Color/Youtube.svg" height="50px" title="YouTube"></a>
<a href="https://tr.ee/-TOCGozNUI" title="Twitter"><img src="https://github.com/gauravghongde/social-icons/blob/master/SVG/Color/Twitter.svg" height="50px"></a>
<a href="https://tr.ee/FlmKJAest5" title="Discord"><img src="https://github.com/gauravghongde/social-icons/blob/master/SVG/Color/Discord.svg" height="50px"></a>

If you find my tools useful please consider supporting via Patreon. I have spent countless hours to make these useful for not only myself but other's as well and am now offering them completely 100% free.

<a href="https://www.patreon.com/kanjieater" rel="nofollow"><img src="https://i.imgur.com/VCTLqLj.png"></a>

If you can't contribute monetarily please consider following on a social platform, joining the discord & sharing a kind message or sharing this with a friend.

# Thanks

Besides the other ones already mentioned & installed this project uses other open source projects subs2cia, & anki-csv-importer

- https://github.com/gsingh93/anki-csv-importer
- https://github.com/kanjieater/subs2cia
- https://github.com/ym1234/audiobooktextsync

# Other Cool Projects

The GOAT delivers again; The best Japanese reading experience ttu-reader paired with SubPlz subs

- https://github.com/Renji-XD/ttu-whispersync
- Demo: https://x.com/kanjieater/status/1834309526129930433

A cool tool to turn these audiobook subs into Visual Novels

- https://github.com/asayake-b5/audiobooksync2renpy