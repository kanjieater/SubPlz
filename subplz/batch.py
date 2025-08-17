# batch.py
from pathlib import Path
from subplz.helpers import extract, rename, copy
from subplz.cli import get_inputs
from subplz.sync import run_sync
from subplz.gen import run_gen
import yaml
from .logger import logger

# A mapping from the command name (str) to the actual function to call.
# This allows us to dynamically execute the correct logic based on the config.
COMMAND_MAP = {
    "rename": rename,
    "extract": extract,
    "sync": run_sync,
    "gen": run_gen,
    "copy": copy,
}


def resolve_pipeline(inputs):
    # This logic only runs if the user did NOT provide a --pipeline on the CLI
    if not inputs.pipeline and inputs.config:
        logger.info(f"Loading pipeline from config file: {inputs.config}")
        try:
            with open(inputs.config, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            loaded_pipeline = config_data.get("batch_pipeline", [])
            inputs.pipeline = loaded_pipeline  # Modify the object in place

            if not loaded_pipeline:
                logger.warning(
                    f"⚠️ 'batch_pipeline' key not found or empty in {inputs.config}"
                )

        except FileNotFoundError:
            logger.error(f"❌ Error: Config file not found at '{inputs.config}'.")
            return None  # Return None on failure
        except yaml.YAMLError as e:
            logger.error(f"❌ Error parsing YAML file '{inputs.config}': {e}.")
            return None  # Return None on failure

    if not inputs.pipeline:
        logger.error(
            "❌ Error: No pipeline to run. Provide one via --pipeline or a valid --config file."
        )
        return None  # Return None if no pipeline is found

    return inputs  # Return the prepared inputs object on success


def run_batch(inputs):
    """
    Prepares and executes a command pipeline on a list of directories.
    This is the main entry point for the batch process.
    """
    # First, resolve the pipeline from the config if necessary.
    prepared_inputs = resolve_pipeline(inputs)

    # If the resolution step failed, stop here.
    if not prepared_inputs:
        logger.error("Batch processing halted due to configuration errors.")
        return

    # Now, the rest of the logic can proceed, confident that inputs.pipeline is populated.
    pipeline = prepared_inputs.pipeline
    directories = prepared_inputs.dirs

    if not directories:
        logger.warning("⚠️ Warning: No directories provided in 'inputs.dirs'. Nothing to do.")
        return

    for dir_string in directories:
        dir_path = Path(dir_string)
        if not dir_path.is_dir():
            logger.warning(f"❗ Skipping invalid directory: {dir_path}")
            continue

        logger.info(f"--- Processing directory: {dir_path} ---")

        for i, step in enumerate(pipeline, 1):
            step_name = step.get("name", f"Step {i}")
            command_template = step.get("command")

            if not command_template:
                logger.warning(f"⚠️ Skipping invalid step (no 'command' found): {step_name}")
                continue

            logger.info(f"\n[{i}/{len(pipeline)}] Executing: {step_name}")

            # Substitute the {directory} placeholder
            args = [
                str(arg).replace("{directory}", str(dir_path))
                for arg in command_template
            ]

            command_name = args[0]
            func_to_call = COMMAND_MAP.get(command_name)

            if not func_to_call:
                logger.error(
                    f"❌ Error: Unknown command '{command_name}' in step '{step_name}'. Skipping."
                )
                continue

            try:
                logger.info(f"  > subplz {' '.join(args)}")
                # The get_inputs function prepares the arguments for the subplz library
                parsed_args = get_inputs(args)
                func_to_call(parsed_args)
            except Exception as e:
                logger.opt(exception=True).error(f"❌ An error occurred during '{step_name}': {e}")
                logger.info("  > Moving to the next step.")

        logger.info(f"\n--- All operations completed for: {dir_path} ---")