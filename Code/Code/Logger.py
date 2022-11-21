import logging


def log(text, indent=0, out=True, debug=True, end=False):
    if debug and out:
        indent = "\t" * indent
        if end:
            print(f"{indent}{text}", end="\t")
        else:
            print(f"{indent}{text}")
    elif out:
        # TODO: Add log file.
        pass
