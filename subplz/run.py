from pathlib import Path

from subplz.transcribe import transcribe
from subplz.sync import sync
from subplz.files import get_chapters, get_streams
from subplz.cache import get_cache
from subplz.models import get_model, set_temperature
from subplz.utils import set_threads
from subplz.cli import get_inputs



def execute_on_inputs():
    inputs = get_inputs()

    set_temperature(inputs.backend)
    set_threads(inputs.backend)
    model = get_model(inputs.backend)
    cache = get_cache(model, inputs.backend, inputs.cache)

    chapters = get_chapters(inputs.sources)
    streams = get_streams(inputs.sources)

    transcribed_streams = transcribe(streams, model, cache, temperature, thread_count, args)

    output_dir = Path(k) if (k := args.pop('output_dir')) else Path('.')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = args.pop('output_format')
    sync(output_dir, output_format, model, transcribed_streams, chapters, cache, temperature, args)


