import os
import subprocess
from multiprocessing import cpu_count
from pathlib import Path
import argparse
from tqdm.contrib.concurrent import process_map
from natsort import os_sorted


def get_mp4_files(audiobook_path):
    mp4_files = os_sorted([str(f) for f in audiobook_path.glob("*.mp4")])
    # print(mp4_files)
    mp4_files = [f.replace(str(audiobook_path), "./") for f in mp4_files]
    mp4_files = [f'"{f}"' for f in mp4_files]
    return mp4_files


def run_docker_cmd_success(audiobook_path, audiobook_name, mp4_files, fallback=False):
    fallback_docker_cmd = (
        f'docker run -it --rm -u $(id -u):$(id -g) -v "{audiobook_path}":/mnt sandreas/m4b-tool:latest merge '
        f"{' '.join(mp4_files)} --output-file \"./{audiobook_name}.m4b\""
    )
    docker_cmd = f"{fallback_docker_cmd} --jobs {cpu_count()}"
    cmd = fallback_docker_cmd if fallback else docker_cmd
    print(cmd)
    try:
        subprocess.run(cmd, shell=True)
        return True
    except Exception as e:
        print(e)
        if fallback:
            return False
        return run_docker_cmd_success(audiobook_path, audiobook_name, mp4_files, True)


def get_m4b_chapters(audiobook_path, audiobook_name):
    try:
        subprocess.check_output(
            f'docker run -it --rm -u $(id -u):$(id -g) -v "{audiobook_path}":/mnt sandreas/m4b-tool:latest chapters '
            f'"./{audiobook_name}.m4b"',
            shell=True,
            text=True,
        ).splitlines()
        return True
    except Exception as e:
        print(e)
        return False


def get_chapter_files(audiobook_path):
    chapter_files = [f for f in audiobook_path.glob("*.chapters.txt")]
    if chapter_files:
        return sorted(chapter_files)[0]


def check_valid_chapters(mp4_files, chapter_file, audiobook_name):
    failed_chapters = []
    if not chapter_file:
        return False
    chapters = []
    # for idx, chapter_file in enumerate(chapter_files):
    with open(chapter_file, "r") as f:
        chapter_info = f.read().splitlines()
    chapters = [line for line in chapter_info if "total-duration" not in line]

    print(f"{audiobook_name}: mp4 {len(mp4_files)}, chapters {len(chapters)}")
    return len(mp4_files) == len(chapters)


def merge_audiobook(audiobook_path):
    audiobook_name = audiobook_path.name
    mp4_files = get_mp4_files(audiobook_path)
    m4b_file = audiobook_path / f"{audiobook_name}.m4b"
    if m4b_file.exists():
        print(f"{audiobook_name}.m4b already exists. Skipping...")
    else:
        success = run_docker_cmd_success(audiobook_path, audiobook_name, mp4_files)
        if not success:
            return audiobook_path

    get_m4b_chapters(audiobook_path, audiobook_name)
    chapter_files = get_chapter_files(audiobook_path)

    if not chapter_files:
        print(f"No chapters file found for {audiobook_name}.")
        return audiobook_path

    valid_chapters = check_valid_chapters(mp4_files, chapter_files, audiobook_name)

    if not valid_chapters:
        return audiobook_path
    else:
        print(f"{audiobook_name} successfully merged and checked!")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audiobooks_path", type=str, help="Path to audiobooks directory"
    )
    args = parser.parse_args()
    audiobooks_path = Path(args.audiobooks_path)

    failed_audiobook_paths = []

    audiobook_paths = [p for p in audiobooks_path.iterdir() if p.is_dir()]

    merge_results = process_map(
        merge_audiobook, audiobook_paths, max_workers=cpu_count(), chunksize=1
    )

    for result in merge_results:
        if result is not None:
            failed_audiobook_paths.append(result)

    if failed_audiobook_paths:
        print("The following audiobooks failed to match chapter names to MP4 files:")
        print("\n".join([str(a) for a in failed_audiobook_paths]))
        with open(audiobooks_path / "failed_audiobooks.txt", "w") as f:
            f.write("\n".join([str(a) for a in failed_audiobook_paths]))
    else:
        # Clear the failed_audiobooks.txt file
        with open(audiobooks_path / "failed_audiobooks.txt", "w") as f:
            pass
        print("All audiobooks successfully merged and checked!")
