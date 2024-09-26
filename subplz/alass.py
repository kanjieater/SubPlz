import subprocess
from pathlib import Path
from subplz.utils import get_tqdm
from subplz.sub import get_subtitle_path, write_subfail

tqdm, trange = get_tqdm()

# Define paths
alass_dir = Path(__file__).parent.parent / 'alass'
alass_path = alass_dir / 'alass-linux64'


def sync_alass(source, input_sources, be):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    with tqdm(source.streams) as bar:
        for batches in bar:
            video_path = batches[2][0].path

            subtitle_path = get_subtitle_path(video_path, input_sources.lang_ext_original)
            target_subtitle_path = get_subtitle_path(video_path, input_sources.lang_ext)
            incorrect_subtitle_path = get_subtitle_path(video_path, input_sources.lang_ext_incorrect)
            if subtitle_path is None and incorrect_subtitle_path is None:
                print(f"❗ Skipping syncing {subtitle_path} since --lang-ext-original and --lang-ext-incorrect were empty")
                continue
            if str(subtitle_path) == str(incorrect_subtitle_path):
                print(f"❗ Skipping syncing {subtitle_path} since the name matches the incorrect timed subtitle")
            if not incorrect_subtitle_path.exists():
                print(f"❗ Subtitle with incorrect timing not found: {incorrect_subtitle_path}")
                continue

            print(f'🤝 Aligning {incorrect_subtitle_path} based on {subtitle_path}')
            cmd = [
                alass_path,
                # *['-' + h for h in alass_args],
                str(subtitle_path),
                str(incorrect_subtitle_path),
                str(target_subtitle_path)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Check if the command was successful
            if result.returncode != 0:
                error_output = result.stderr
                stdout_output = result.stdout
                error_message = (
                    f"❗ Alass command failed: {result.returncode}\n\n"
                    f"stdout: {stdout_output}\n\n"
                    f"Error output: {error_output}"
                )
                print(error_message)
                write_subfail(source, target_subtitle_path, error_message)
            else:
                # If successful, proceed with your logic here
                source.writer.written = True
