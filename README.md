# Approximate Data Profiling -- Detecting Approximate INDs

This tool allows you to find INDs on datasets using [Metanome](https://metanome.de).

## Setup

To run this tool, Python is required. We haven't verified exact version compatibility, but anything 3.9+ should be fine.
We recommend a virtual environment like [venv](https://docs.python.org/3/library/venv.html).
We now assume you have a suitable Python version running as `python3` and the corresponding PIP as `pip3` (note that this can vary between distributions, make sure to replace those in the following commands).
All commands are expected to be run from the repo's root directory (where this file is situated).

1. Install requirements: `pip3 install -r requirements.txt`.
2. Have a copy of [Metanome CLI](https://github.com/HPI-Information-Systems/Metanome/tree/metanome_cli/metanome-cli) as `metanome-cli.jar`.
3. Have a copy of [BINDER](https://github.com/HPI-Information-Systems/metanome-algorithms/tree/master/BINDER) as `BINDER.jar`.
4. Have the following (empty) directories: `output`, `results`, `src`, `tmp`. You can also specify other paths when running the tool, but these are the default values.
5. Place the data to be sampled in `src` as `csv` files (no subdirectories).
6. Sample the data and run the experiments: `python3 -m pysrc.scripts.sampling`. It supports `-h` to list all arguments.
7. Evaluate the results: `python3 -m pysrc.scripts.evaluation --file something.json` where `something.json` is the path to a file generated by the sampling step. This also supports `-h`.
