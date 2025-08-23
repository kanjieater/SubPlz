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
        "jap": "ja",
        "ger": "de",
        "fre": "fr",
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
    print(f"❗Language code '{lang_code_input}' not recognized by pycountry.")
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


def get_host_path(config, path_from_job):
    """
    Translates a container path to a host path, or confirms a host path's existence.
    """
    path_map = config.get("watcher", {}).get("path_map", {})
    str_path_from_job = str(path_from_job)

    if str_path_from_job in path_map:
        return path_map[str_path_from_job]

    for docker_path, host_path in path_map.items():
        if str_path_from_job.startswith(docker_path):
            return str_path_from_job.replace(docker_path, host_path, 1)

    if os.path.exists(str_path_from_job):
        return str_path_from_job

    return str_path_from_job


def get_docker_path(config, path_from_host):
    """
    Translates a host path back to a container path.
    """
    path_map = config.get("watcher", {}).get("path_map", {})
    sorted_host_paths = sorted(path_map.values(), key=len, reverse=True)

    str_path_from_host = str(path_from_host)
    for host_path in sorted_host_paths:
        if str_path_from_host.startswith(host_path):
            for docker_path, mapped_host_path in path_map.items():
                if mapped_host_path == host_path:
                    return str_path_from_host.replace(host_path, docker_path, 1)
    return str_path_from_host


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
