import sys

# Source: https://stackoverflow.com/a/14981125
def eprint(*args, **kwargs):
    """Print to stderr. Same arguments as built-in print."""
    return print(*args, file=sys.stderr, **kwargs)
