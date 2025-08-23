from pathlib import Path
import shlex
from .logger import logger
from .helpers import extract, rename, copy
from .cli import get_inputs, get_args
from .sync import run_sync
from .gen import run_gen

COMMAND_MAP = {
    "rename": rename,
    "extract": extract,
    "sync": run_sync,
    "gen": run_gen,
    "copy": copy,
}


def run_batch(inputs):
    """
    Prepares and executes a command pipeline. It will attempt all steps
    and report a final failure if any single step failed.
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
            except ValueError as e:
                logger.error(
                    f"❌ Error parsing command in step '{step_name}': {e}. Check for unclosed quotes."
                )
                continue

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
                continue

            try:
                logger.log("CMD", f"  > subplz {' '.join(final_args_list)}")
                sub_args = get_args(final_args_list)
                structured_inputs = get_inputs(sub_args, config)
                step_was_successful = func_to_call(structured_inputs)

                if not step_was_successful:
                    logger.error(f"❌ Step '{step_name}' reported a failure. Continuing to next step.")
                    pipeline_overall_success = False # Mark the whole job as failed

            except Exception as e:
                logger.opt(exception=True).error(
                    f"❌ A critical error occurred during '{step_name}': {e}"
                )
                pipeline_overall_success = False # Mark the whole job as failed
                logger.info("  > Continuing to the next step.")

        if not pipeline_overall_success:
            raise Exception(f"One or more steps in the batch pipeline failed for '{dir_path.name}'.")

        logger.info(f"\n--- All operations completed successfully for: {dir_path.name} ---")