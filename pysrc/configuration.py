from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
import datetime
from typing import Any, TypeVar

from dateutil import parser as datetime_parser


@dataclass(frozen=True)
class GlobalConfiguration:
    algorithm: str
    arity: str
    sampling_rates: list[float]
    sampling_methods: list[str]

    header: bool
    clip_output: bool
    print_inds: bool
    create_plots: bool

    now: datetime.datetime

    source_dir: str
    tmp_folder: str
    results_folder: str
    results_suffix: str
    output_folder: str
    output_file: str
    plot_folder: str

    T = TypeVar('T')

    @staticmethod
    def _construct_from_default(args: dict[str, Any], key: str, required_type: type, default: T) -> T:
        return args[key] if key in args and isinstance(args[key], required_type) else default

    @classmethod
    def default(cls, args: dict[str, Any]):
        algorithm = cls._construct_from_default(args, 'algorithm', str, ['BINDER', 'PartialSPIDER'][1])
        arity = cls._construct_from_default(args, 'arity', str, ['unary', 'nary'][0])
        now = cls._construct_from_default(args, 'now', datetime.datetime, datetime.datetime.now())
        now_date = f'{now.year}{now.month:02d}{now.day}'
        now_time = f'{now.hour}{now.minute:02d}{now.second:02d}'
        return cls(
            algorithm=algorithm,
            arity=arity,
            sampling_rates=[0.1, 0.01, 0.001],
            sampling_methods=['random', 'first', 'evenly-spaced'],
            header=cls._construct_from_default(args, 'header', bool, False),
            clip_output=cls._construct_from_default(args, 'clip_output', bool, True),
            print_inds=cls._construct_from_default(args, 'print_inds', bool, False),
            create_plots=cls._construct_from_default(args, 'create_plots', bool, True),
            now=now,
            source_dir=cls._construct_from_default(args, 'source_dir', str, 'src/'),
            tmp_folder=cls._construct_from_default(args, 'tmp_folder', str, 'tmp/'),
            results_folder=cls._construct_from_default(args, 'results_folder', str, 'results/'),
            results_suffix=cls._construct_from_default(args, 'results_suffix', str, '_inds'),
            output_folder=cls._construct_from_default(args, 'output_folder', str, 'output/'),
            output_file=cls._construct_from_default(args, 'output_file', str, f'output_{arity}_{now_date}_{now_time}'),
            plot_folder=cls._construct_from_default(args, 'plot_folder', str, 'plots/'),
        )
        
    @staticmethod
    def argparse_arguments(parser: ArgumentParser) -> None:
        """Modifies the provided argparse parser by adding optional arguments for all config options"""
        parser.add_argument('--algorithm', type=str, required=False, default=None, help='Which algorithm to use, either PartialSPIDER or BINDER')
        parser.add_argument('--arity', type=str, required=False, default=None, help='Whether to find `unary` or `nary` INDs')
        parser.add_argument('--now', type=datetime_parser.parse, required=False, default=None, help='The time to use as `now`')
        parser.add_argument('--header', action=BooleanOptionalAction, required=False, default=None, help='Whether the provided csv files have headers')
        parser.add_argument('--clip-output', action=BooleanOptionalAction, required=False, default=None, help='Whether to clip the Metanome CLI output to onyl show the runtime')
        parser.add_argument('--print-inds', action=BooleanOptionalAction, required=False, default=None, help='Whether to print the found INDs')
        parser.add_argument('--create-plots', action=BooleanOptionalAction, required=False, default=None, help='Whether to create plots')
        parser.add_argument('--source-dir', type=str, required=False, default=None, help='The directory containing the data files to sample and experiment on')
        parser.add_argument('--tmp-folder', type=str, required=False, default=None, help='The directory containing temporary files while the program runs')
        parser.add_argument('--results-folder', type=str, required=False, default=None, help='The directory containing temporary result files while the program runs')
        parser.add_argument('--results-suffix', type=str, required=False, default=None, help='What to append to the result files')
        parser.add_argument('--output-folder', type=str, required=False, default=None, help='The directory containing result files after the program terminates')
        parser.add_argument('--output-file', type=str, required=False, default=None, help='The name of the file that contains the output, without folder and extension')
        parser.add_argument('--plot-folder', type=str, required=False, default=None, help='The directory containing result plots after the program terminates')
