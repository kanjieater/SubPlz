import ffmpeg
import subprocess
from pathlib import Path
from subplz.utils import get_tqdm
from subplz.files import normalize_text

tqdm, trange = get_tqdm()

# Define paths
alass_dir = Path(__file__).parent.parent / 'alass'
alass_path = alass_dir / 'alass-linux64'


def extract_subtitles(video_path: Path, output_subtitle_path: Path) -> None:
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(output_subtitle_path), map='0:s:0', c='srt', loglevel="quiet")
            .global_args("-hide_banner")
            .run(overwrite_output=True)
        )
        return output_subtitle_path
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to extract subtitles: {e.stderr.decode()}\nCommand: {str(e.cmd)}")

# Define function to run alass
def sync_alass(source, input_sources, be):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    with tqdm(source.streams) as bar:
        for batches in bar:
            video_path = batches[2][0].path
            subtitle_path = Path(video_path).parent / f"{Path(video_path).stem}.{input_sources.lang_ext_original}.srt"
            incorrect_subtitle_path = Path(video_path).parent / f"{Path(video_path).stem}.{input_sources.lang_ext_incorrect}.srt"
            target_subtitle_path = Path(video_path).parent / f"{Path(video_path).stem}.{input_sources.lang_ext}.srt"

            bar.set_description(f'Extracting subtitles from {video_path}')

            # Extract subtitles with ffmpeg
            if not subtitle_path.exists():
                extract_subtitles(video_path, subtitle_path)

            # Run alass alignment
            bar.set_description(f'Aligning {batches[0]}')
            cmd = [
                alass_path,
                # *['-' + h for h in alass_args],
                str(subtitle_path),
                str(incorrect_subtitle_path),
                str(target_subtitle_path)
            ]
            try:
                subprocess.run(cmd)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Alass command failed: {e.stderr.decode()}\n args: {' '.join(cmd)}") from e
