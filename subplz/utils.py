from glob import glob, escape
from pathlib import Path
from functools import partialmethod
import torch
from natsort import os_sorted


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell" or shell == "google.colab._shell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return True  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_tqdm(progress=True):
    t = None
    if is_notebook():
        from tqdm.notebook import tqdm, trange

        t = tqdm
    else:
        from tqdm import tqdm, trange

        t = tqdm
    t.__init__ = partialmethod(tqdm.__init__, disable=not progress)
    trange.__init__ = partialmethod(trange.__init__, disable=not progress)

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
