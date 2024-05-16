from pathlib import Path

from files import get_working_folders
from transcribe import transcribe
from sync import sync
from files import get_chapters, get_streams
from cache import get_cache
from models import get_model, get_temperature
from utils import get_thread_count
from cli import get_args


def get_inputs():
    args = get_args()
    print(args)
    working_folders = get_working_folders(args['dirs'])
    print(working_folders)
    return args


def execute_on_inputs():
    args = get_inputs()

    temperature = get_temperature(args)
    thread_count = get_thread_count(args)
    model = get_model(args, thread_count)
    cache = get_cache(args, model)

    chapters = get_chapters(args)
    streams = get_streams(args)

    transcribed_streams = transcribe(streams, model, cache, temperature, thread_count, args)

    output_dir = Path(k) if (k := args.pop('output_dir')) else Path('.')#os.path.dirname(args['audio'][0]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = args.pop('output_format')
    sync(output_dir, output_format, model, transcribed_streams, chapters, cache, temperature, args)


if __name__ == "__main__":
    execute_on_inputs()
