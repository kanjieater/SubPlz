from subplz.transcribe import transcribe
from subplz.sync import sync
from subplz.gen import gen
from subplz.files import get_sources, post_process
from subplz.models import get_model, get_temperature
from subplz.utils import get_threads
from subplz.cli import get_inputs
from subplz.utils import get_tqdm

tqdm, trange = get_tqdm()


def execute_on_inputs():
    inputs = get_inputs()
    be = inputs.backend

    be.temperature = get_temperature(be)
    be.threads = get_threads(be)
    model = get_model(be)

    sources = get_sources(inputs.sources, inputs.cache)
    for source in tqdm(sources):
        print(f"🐼 Starting '{source.audio}'...")
        transcribed_streams = transcribe(source.streams, model, be)
        if inputs.subcommand == "sync":
            sync(
                source,
                model,
                transcribed_streams,
                be,
            )
        elif inputs.subcommand == "gen":
            gen(
                source,
                model,
                transcribed_streams,
                be,
            )
    post_process(sources, inputs.subcommand)
