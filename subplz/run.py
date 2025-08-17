from subplz.helpers import find, rename, copy, extract
from subplz.batch import run_batch
from subplz.sync import run_sync
from subplz.gen import run_gen
from subplz.cli import get_inputs
from subplz.watcher import run_watcher

def execute_on_inputs():
    """
    Parses CLI arguments and dispatches to the correct handler function.
    """
    inputs = get_inputs()
    COMMAND_MAP = {
        "watch": run_watcher,
        "find":    lambda args: find(args.dirs),
        "rename":  rename,
        "copy":    copy,
        "extract": extract,
        "batch":   run_batch,
        "sync":    run_sync,
        "gen":     run_gen,
    }

    handler_function = COMMAND_MAP.get(inputs.subcommand)

    if handler_function:
        handler_function(inputs)
    else:
        print(f"Error: Unknown subcommand '{inputs.subcommand}'")