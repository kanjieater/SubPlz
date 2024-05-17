from pathlib import Path
from subplz.transcribe import transcribe
from subplz.sync import sync
from subplz.files import get_chapters, get_streams
from subplz.cache import get_cache
from subplz.models import get_model, get_temperature
from subplz.utils import get_threads
from subplz.cli import get_inputs


def execute_on_inputs():
    inputs = get_inputs()
    be = inputs.backend

    be.temperature = get_temperature(be)
    be.threads = get_threads(be)
    model = get_model(be)
    cache = get_cache(model, be, inputs.cache)

    chapters = get_chapters(inputs.sources.text)
    streams = get_streams(inputs.sources.audio)

    transcribed_streams = transcribe(streams, model, cache, be)

    output_dir = Path(k) if (k := args.pop("output_dir")) else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = args.pop("output_format")
    sync(
        output_dir,
        output_format,
        model,
        transcribed_streams,
        chapters,
        cache,
        temperature,
        args,
    )
