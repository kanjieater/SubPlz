from pathlib import Path
from subplz.transcribe import transcribe
from subplz.sync import sync
from subplz.files import get_chapters, get_streams, get_sources
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
    cache = get_cache(be, inputs.cache)

    sources = get_sources(inputs.sources)
    for source in sources:
        chapters = get_chapters(source.text)
        streams = get_streams(source.audio)
        transcribed_streams = transcribe(streams, model, cache, be)
        sync(
            source,
            model,
            transcribed_streams,
            chapters,
            cache,
            be,
        )
