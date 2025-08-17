from glob import glob, escape
from pathlib import Path
from functools import partialmethod
import torch
from natsort import os_sorted
import pycountry
from subplz.logger import TqdmToLogger


def get_iso639_2_lang_code(lang_code_input: str) -> str | None:
    # ... (as defined before) ...
    if not lang_code_input:
        return None
    lang_code_input = lang_code_input.lower()
    try:
        lang = pycountry.languages.get(alpha_2=lang_code_input)
        if lang:
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

    # --- KEY CHANGE #3: Force all tqdm instances to use our custom stream ---
    # This automatically directs tqdm's output to our "TQDM" log level.
    # `file=tqdm_stream` replaces the default `file=sys.stderr`.
    # `dynamic_ncols=True` ensures the bar resizes with the terminal window.
    t.__init__ = partialmethod(
        tqdm.__init__, disable=not progress, file=tqdm_stream, dynamic_ncols=True
    )
    trange.__init__ = partialmethod(
        trange.__init__, disable=not progress, file=tqdm_stream, dynamic_ncols=True
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
