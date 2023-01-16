import os

def is_non_zero_file(fpath: str) -> bool:
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
