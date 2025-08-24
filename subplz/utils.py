import os
from glob import glob, escape
from pathlib import Path
from functools import partialmethod
import torch
from .logger import TqdmToLogger, logger
from natsort import os_sorted
import pycountry


def get_lang_code(lang_code_input: str) -> str | None:
    if not lang_code_input:
        return None
    lang_code_input = lang_code_input.lower()
    MANUAL_LANG_FIXES = {
        # Common Abbreviations
        "jap": "ja",
        "jp": "ja",

        # Old ISO 639-2/B (Bibliographic) codes to new ISO 639-1 (2-letter) codes
        "chi": "zh",
        "ger": "de",
        "fre": "fr",
        "dut": "nl",
        "gre": "el",
        "ice": "is",
        "mac": "mk",
        "mao": "mi",
        "may": "ms",
        "per": "fa",
        "tib": "bo",
        "wel": "cy",
        "alb": "sq",
        "arm": "hy",
        "baq": "eu",
        "bur": "my",
        "cze": "cs",
        "geo": "ka",
        "slo": "sk",

        # Common full language names
        "english": "en",
        "japanese": "ja",
        "chinese": "zh",
        "german": "de",
        "french": "fr",
        "spanish": "es",
        "korean": "ko",
        "portuguese": "pt",
    }
    if lang_code_input in MANUAL_LANG_FIXES:
        lang_code_input = MANUAL_LANG_FIXES[lang_code_input]
    try:
        lang = pycountry.languages.get(alpha_2=lang_code_input)
        if lang:
            # Prefer the bibliographic (T) code like 'jpn' over the terminology (B) code
            return getattr(lang, "bibliographic", getattr(lang, "alpha_3", None))
    except KeyError:
        pass
    try:
        lang = pycountry.languages.get(alpha_3=lang_code_input)
        if lang:
            return getattr(lang, "bibliographic", getattr(lang, "alpha_3", None))
    except KeyError:
        pass
    logger.warning(f"❗Language code '{lang_code_input}' not recognized")
    return None


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell" or shell == "google.colab._shell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return True
    except NameError:
        return False


def get_tqdm(progress=True):
    t = None
    if is_notebook():
        from tqdm.notebook import tqdm, trange

        t = tqdm
    else:
        from tqdm import tqdm, trange

        t = tqdm

    # --- KEY CHANGE #2: Create an instance of our TQDM stream ---
    tqdm_stream = TqdmToLogger()

    # Define a log-friendly bar format
    # This format prints a new line for each update, which is ideal for log files.
    # It removes the dynamic bar drawing characters.
    log_friendly_format = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )

    # --- KEY CHANGE: Force tqdm to behave correctly in non-TTY environments ---
    t.__init__ = partialmethod(
        tqdm.__init__,
        disable=not progress,
        file=tqdm_stream,
        dynamic_ncols=True,
        # Force ASCII characters, which is safer for logs
        ascii=True,
        # Use a format that doesn't rely on carriage returns
        bar_format=log_friendly_format,
    )
    trange.__init__ = partialmethod(
        trange.__init__,
        disable=not progress,
        file=tqdm_stream,
        dynamic_ncols=True,
        ascii=True,
        bar_format=log_friendly_format,
    )

    return t, trange


def get_threads(inputs):
    threads = inputs.threads
    if threads > 0:
        torch.set_num_threads(threads)
    return threads


def grab_files(folder, types, sort=True):
    files = []
    for t in types:
        pattern = f"{escape(folder)}/{t}"
        files.extend(glob(pattern))
    if sort:
        return os_sorted(files)
    return files


def get_tmp_path(file_path):
    file_path = Path(file_path)
    filename = file_path.stem
    return file_path.parent / f"{filename}.tmp{file_path.suffix}"

def get_host_path(config, docker_path):
    """
    Translates a canonical Docker path to its corresponding host path using the path_map.
    (Handles the new 'host: container' config format).
    """
    path_map = config.get("watcher", {}).get("path_map", {})
    str_docker_path = str(docker_path)

    # Sort items by the length of the container path (value) to handle nested paths
    sorted_map_items = sorted(path_map.items(), key=lambda item: len(item[1]), reverse=True)

    for host_path, container_path in sorted_map_items:
        if str_docker_path.startswith(container_path):
            return str_docker_path.replace(container_path, host_path, 1)

    if os.path.exists(str_docker_path):
        return str_docker_path

    return str_docker_path

def get_docker_path(config, host_path):
    """
    Translates a host path back to its canonical Docker path for writing into job files.
    (Handles the new 'host: container' config format).
    """
    path_map = config.get("watcher", {}).get("path_map", {})
    str_host_path = str(host_path)

    # Sort items by the length of the host path (key) to handle nested paths
    sorted_map_items = sorted(path_map.items(), key=lambda item: len(item[0]), reverse=True)

    for h_path, container_path in sorted_map_items:
        if str_host_path.startswith(h_path):
            return str_host_path.replace(h_path, container_path, 1)

    return str_host_path


def resolve_local_path(config, any_path):
    """
    Resolves any given path (host or container) to a locally accessible path,
    regardless of the execution environment. This is the key to full environment independence.

    Args:
        config (dict): The full application configuration.
        any_path (str or Path): A path that could be in either host or container format.

    Returns:
        str: A string representing the path that is physically accessible from the
             current environment.
    """
    if not any_path:
        return None

    path_str = str(any_path)

    # Candidate 1: Assume `any_path` is a container path and find its host equivalent.
    potential_host_path = get_host_path(config, path_str)

    # Candidate 2: Assume `any_path` is a host path and find its container equivalent.
    potential_docker_path = get_docker_path(config, path_str)

    # --- The Core Logic ---
    # Now, check which of our calculated paths actually exists in the current environment.
    # This order of checks is robust for all scenarios.

    # If running on the HOST, this will almost always be the correct one.
    if os.path.exists(potential_host_path):
        return potential_host_path

    # If running in the CONTAINER, this will be the correct one.
    # This also handles the user's specific `docker exec` case.
    elif os.path.exists(potential_docker_path):
        return potential_docker_path

    # Fallback for unmapped paths. If neither translation exists,
    # maybe the original path was correct all along (e.g. "/tmp/").
    elif os.path.exists(path_str):
        return path_str

    # If nothing is found, we must return one for error reporting.
    # We'll prefer the path that a translation actually occurred on.
    # If the user provided a host path, the docker path is the likely target inside a container.
    if path_str != potential_docker_path:
        return potential_docker_path
    # Otherwise, default to the host path.
    return potential_host_path

def find_and_show_lingering_tensors():
    """
    Forces garbage collection and scans for any PyTorch tensors still on the GPU.
    This is a powerful debugging tool for VRAM leaks.
    """
    import gc
    import torch

    logger.debug("--- Running VRAM Leak Diagnostics ---")
    # Force a garbage collection to clean up any easy-to-find unreferenced objects
    gc.collect()

    lingering_tensors_found = 0
    total_size_mb = 0

    # Scan all objects tracked by the garbage collector
    for obj in gc.get_objects():
        try:
            # Check if the object is a PyTorch tensor and if it's on the CUDA device
            if torch.is_tensor(obj) and obj.is_cuda:
                lingering_tensors_found += 1
                tensor_size_mb = obj.nelement() * obj.element_size() / 1024**2
                total_size_mb += tensor_size_mb

                logger.warning(
                    f"Found lingering tensor: size={list(obj.size())}, "
                    f"memory={tensor_size_mb:.2f} MB, dtype={obj.dtype}"
                )

                # If you have objgraph installed (pip install objgraph), this is the magic bullet:
                # It will show you exactly what variable is holding onto the tensor.
                # import objgraph
                # logger.warning("Back-references (what is holding the tensor):")
                # objgraph.show_backrefs([obj], max_depth=5)

        except Exception:
            # Some objects may not support these checks
            continue

    if lingering_tensors_found == 0:
        logger.debug(
            "✅ No lingering CUDA tensors found. The leak may be in the C++ backend."
        )
    else:
        logger.warning(
            f"Found {lingering_tensors_found} total lingering tensors, consuming ~{total_size_mb:.2f} MB directly."
        )

    logger.debug("--- End of VRAM Leak Diagnostics ---")
