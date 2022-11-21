import logging


def log(text, indent=0, out=True, debug=True):
    if debug and out:
        indent = "\t" * indent
        print(f"{indent}{text}")
    elif out:
        # TODO: Add log file.
        pass
