from pathlib import Path

from subplz.transcribe import transcribe
from subplz.sync import sync
from subplz.files import get_chapters, get_streams
from subplz.cache import get_cache
from subplz.models import get_model, get_temperature
from subplz.utils import get_thread_count
from subplz.cli import get_inputs



def execute_on_inputs():
    inputs = get_inputs()

    temperature = get_temperature(args)
    thread_count = get_thread_count(args)
    model = get_model(args, thread_count)
    cache = get_cache(args, model)

    chapters = get_chapters(args)
    streams = get_streams(args)

    transcribed_streams = transcribe(streams, model, cache, temperature, thread_count, args)

    output_dir = Path(k) if (k := args.pop('output_dir')) else Path('.')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = args.pop('output_format')
    sync(output_dir, output_format, model, transcribed_streams, chapters, cache, temperature, args)


