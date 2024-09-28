import subprocess
from pathlib import Path
from subplz.utils import get_tqdm
from subplz.sub import get_subtitle_path, write_subfail, sanitize_subtitle, convert_between_sub_format

tqdm, trange = get_tqdm()

# Define paths
alass_dir = Path(__file__).parent
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
            temp_srt_path = ''
            temp_incorrect_srt_path = ''
            try:
                sanitize_subtitle(subtitle_path)
                sanitize_subtitle(incorrect_subtitle_path)
                if subtitle_path.suffix != incorrect_subtitle_path.suffix:
                    temp_srt_path = subtitle_path.with_suffix('.tmp.srt')
                    temp_incorrect_srt_path = incorrect_subtitle_path.with_suffix('.tmp.srt')

                    print(f"🔄 Converting {subtitle_path} and {incorrect_subtitle_path} to SRT format for Alass")
                    convert_between_sub_format(str(subtitle_path), str(temp_srt_path), format='srt')
                    convert_between_sub_format(str(incorrect_subtitle_path), str(temp_incorrect_srt_path), format='srt')
                    subtitle_path = temp_srt_path
                    incorrect_subtitle_path = temp_incorrect_srt_path
            except Exception as err:
                error_message = (
                    f"""❗ Failed to Sanitize subtitles; {subtitle_path}
                    {err}
                    """
                )
                print(error_message)
                write_subfail(source, target_subtitle_path, error_message)

            print(f'🤝 Aligning {incorrect_subtitle_path} based on {subtitle_path}')
            cmd = [
                alass_path,
                # *['-' + h for h in alass_args],
                str(subtitle_path),
                str(incorrect_subtitle_path),
                str(target_subtitle_path)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

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
                source.writer.written = True

            if temp_srt_path and temp_srt_path.exists():
                temp_srt_path.unlink()

            if temp_incorrect_srt_path and temp_incorrect_srt_path.exists():
                temp_incorrect_srt_path.unlink()
