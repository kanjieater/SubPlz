import sys
import multiprocessing
import shlex
from pathlib import Path

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
    "sync": "run_sync",
    "gen": "run_gen",
    "copy": copy,
}

# MINIMAL CHANGE: Worker now accepts a command name and lazy-imports AI functions.
def _step_worker(command_name, inputs, config):
    try:
        configure_logging(config)

        func_to_call = None
        if command_name == "sync":
            from .sync import run_sync
            func_to_call = run_sync
        elif command_name == "gen":
            from .gen import run_gen
            func_to_call = run_gen
        else:
            func_to_call = COMMAND_MAP[command_name]

        step_was_successful = func_to_call(inputs)
        sys.exit(0 if step_was_successful is not False else 1)

    except Exception:
        logger.opt(exception=True).error(
            "A critical error occurred inside the step worker."
        )
        sys.exit(1)


def run_batch(inputs):
    config = inputs.config_data
    pipeline = config.get("batch_pipeline", [])
    directories = inputs.dirs
    target_file = inputs.file

    watcher_settings = config.get("watcher", {})
    job_timeout = watcher_settings.get("job_timeout_seconds", 600)

    if not pipeline:
        logger.error("No 'batch_pipeline' found in config. Nothing to do.")
        return
    if not directories:
        logger.warning("No directories provided to batch command. Nothing to do.")
        return

    for dir_string in directories:
        dir_path = Path(dir_string)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"The directory specified in the job does not exist or is not a directory: {dir_path}")

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

            process = None
            try:
                args_list = shlex.split(final_command_string)
                final_args_list = [arg for arg in args_list if arg]

                if not final_args_list:
                    logger.warning(
                        f"Skipping step '{step_name}': Command is empty after processing."
                    )
                    continue

                command_name = final_args_list[0]
                if command_name not in COMMAND_MAP:
                    logger.error(
                        f"❌ Error: Unknown command '{command_name}' in step '{step_name}'. Skipping."
                    )
                    pipeline_overall_success = False
                    continue

                sub_args = get_args(final_args_list)
                structured_inputs = get_inputs(sub_args, config)
                structured_inputs.config_data = config

                logger.log("CMD", f"  > subplz {' '.join(final_args_list)}")

                ctx = multiprocessing.get_context("spawn")
                # MINIMAL CHANGE: Pass the command_name string instead of the function object.
                process = ctx.Process(
                    target=_step_worker, args=(command_name, structured_inputs, config)
                )
                process.start()

                process.join(timeout=job_timeout)

                if process.is_alive():
                    logger.critical(f"🚨 Step '{step_name}' timed out after {job_timeout} seconds. Terminating.")
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
                    pipeline_overall_success = False
                    logger.error(f"❌ Step '{step_name}' failed due to a timeout.")
                    continue

                if process.exitcode != 0:
                    pipeline_overall_success = False
                    logger.error(
                        f"❌ Step '{step_name}' failed. Continuing to the next step."
                    )

            except Exception as e:
                logger.opt(exception=True).error(
                    f"❌ A critical error occurred while preparing step '{step_name}': {e}"
                )
                pipeline_overall_success = False
                logger.info("  > Continuing to the next step.")
            finally:
                if process and process.is_alive():
                    process.kill()
                    process.join()


        if not pipeline_overall_success:
            raise Exception(
                f"One or more steps in the batch pipeline failed for '{dir_path.name}'."
            )

        logger.success(f"\n--- All operations completed for: {dir_path.name} ---")