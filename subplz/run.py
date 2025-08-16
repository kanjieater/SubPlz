
from subplz.helpers import find, rename, copy, extract
from subplz.batch import run_batch
from subplz.sync import run_sync
from subplz.gen import run_gen
from subplz.cli import get_inputs


def execute_on_inputs():
    inputs = get_inputs()
    if inputs.subcommand == "find":
        find(inputs.dirs)
        return
    if inputs.subcommand == "rename":
        rename(inputs)
        return
    if inputs.subcommand == "copy":
        copy(inputs)
        return
    if inputs.subcommand == "extract":
        extract(inputs)
        return
    if inputs.subcommand == "batch":
        run_batch(inputs)
        return
    if inputs.subcommand == "sync":
        run_sync(inputs)
    elif inputs.subcommand == "gen":
        run_gen(inputs)
