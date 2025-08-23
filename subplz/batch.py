import sys
import multiprocessing
import shlex
from pathlib import Path

# We need to import configure_logging to set it up in each new process
from .logger import logger, configure_logging
from .helpers import extract, rename, copy
from .cli import get_inputs, get_args
from .sync import run_sync
from .gen import run_gen

# This script is now the sole manager of process isolation for pipeline steps.
# It is called directly by the watcher.

COMMAND_MAP = {
    "rename": rename,
    "extract": extract,
    "sync": run_sync,
    "gen": run_gen,
    "copy": copy,
}


def _step_worker(func_to_call, inputs, config):
    """
    Worker target for a single pipeline step.
    Configures its own logger and translates the function's boolean result
    into a process exit code.
    """
    try:
        # Each new process must configure its own logger to get custom levels
        configure_logging(config)

        # Execute the actual command function (e.g., rename, extract)
        step_was_successful = func_to_call(inputs)

        # A function returning False is a non-critical failure; exit with 1
        if step_was_successful is False:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception:
        # A crash/exception is a critical failure; exit with 1
        logger.opt(exception=True).error(
            "A critical error occurred inside the step worker."
        )
        sys.exit(1)


def run_batch(inputs):
    """
    Prepares and executes a command pipeline. Each step is run in an
    isolated process. It will attempt all steps and report a final failure
    if any single step failed.
    """
    config = inputs.config_data
    pipeline = config.get("batch_pipeline", [])
    directories = inputs.dirs
    target_file = inputs.file

    if not pipeline:
        logger.error("No 'batch_pipeline' found in config. Nothing to do.")
        return
    if not directories:
        logger.warning("No directories provided to batch command. Nothing to do.")
        return

    for dir_string in directories:
        dir_path = Path(dir_string)
        if not dir_path.is_dir():
            logger.warning(f"❗ Skipping invalid directory: {dir_path}")
            continue

        logger.info(f"--- Processing directory: {dir_path.name} ---")

        pipeline_overall_success = True

        for i, step in enumerate(pipeline, 1):
            step_name = step.get("name", f"Step {i}")
            command_string_template = step.get("command")

            if not isinstance(command_string_template, str):
                logger.warning(
                    f"⚠️ Skipping invalid step '{step_name}': 'command' must be a string."
                )
                continue

            logger.info(f"\n[{i}/{len(pipeline)}] Executing: {step_name}")

            final_command_string = command_string_template.replace(
                "{directory}", str(dir_path)
            ).replace("{file}", str(target_file) if target_file else "")

            try:
                args_list = shlex.split(final_command_string)
                final_args_list = [arg for arg in args_list if arg]

                if not final_args_list:
                    logger.warning(
                        f"Skipping step '{step_name}': Command is empty after processing."
                    )
                    continue

                command_name = final_args_list[0]
                func_to_call = COMMAND_MAP.get(command_name)

                if not func_to_call:
                    logger.error(
                        f"❌ Error: Unknown command '{command_name}' in step '{step_name}'. Skipping."
                    )
                    pipeline_overall_success = False
                    continue

                # Prepare inputs for the subprocess
                sub_args = get_args(final_args_list)
                structured_inputs = get_inputs(sub_args, config)

                logger.log("CMD", f"  > subplz {' '.join(final_args_list)}")

                # Spawn the isolated process for the step
                ctx = multiprocessing.get_context("spawn")
                process = ctx.Process(
                    target=_step_worker, args=(func_to_call, structured_inputs, config)
                )
                process.start()
                process.join()

                # Check the result from the process's exit code
                if process.exitcode != 0:
                    pipeline_overall_success = False
                    logger.error(
                        f"❌ Step '{step_name}' failed. Continuing to the next step."
                    )

            except Exception as e:
                # This outer catch is for errors during argument parsing or process creation
                logger.opt(exception=True).error(
                    f"❌ A critical error occurred while preparing step '{step_name}': {e}"
                )
                pipeline_overall_success = False
                logger.info("  > Continuing to the next step.")

        if not pipeline_overall_success:
            raise Exception(
                f"One or more steps in the batch pipeline failed for '{dir_path.name}'."
            )

        logger.success(f"\n--- All operations completed for: {dir_path.name} ---")
