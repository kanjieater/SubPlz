def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell' or shell == 'google.colab._shell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return True  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def get_tqdm():
    if is_notebook():
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange
    return tqdm