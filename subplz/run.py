from subplz.transcribe import transcribe
from subplz.sync import sync
from subplz.alass import sync_alass
from subplz.gen import gen
from subplz.helpers import find, rename, copy
from subplz.files import get_sources, post_process
from subplz.models import get_model, get_temperature
from subplz.utils import get_threads
from subplz.cli import get_inputs
from subplz.utils import get_tqdm

tqdm, trange = get_tqdm()


def execute_on_inputs():
    inputs = get_inputs()
    if inputs.subcommand == "find":
        find()
    if inputs.subcommand == "rename":
        rename()
    if inputs.subcommand == "copy":
        copy()

    be = inputs.backend

    be.temperature = get_temperature(be)
    be.threads = get_threads(be)
    model = get_model(be)

    sources = get_sources(inputs.sources, inputs.cache)
    for source in tqdm(sources):
        print(f"🐼 Starting '{source.audio}'...")

        if inputs.subcommand == "sync" and inputs.backend.alass:
            sync_alass(source, inputs.sources, be)
        elif inputs.subcommand == "sync":
            transcribed_streams = transcribe(source.streams, model, be)
            sync(
                source,
                model,
                transcribed_streams,
                be,
            )
        elif inputs.subcommand == "gen":
            transcribed_streams = transcribe(source.streams, model, be)
            gen(
                source,
                model,
                transcribed_streams,
                be,
            )
    post_process(sources, inputs.subcommand, inputs.backend.alass, inputs.sources.lang_ext_original)
