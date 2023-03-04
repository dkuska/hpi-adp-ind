from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
import datetime
from typing import Any, Optional, Type, TypeVar

from dateutil import parser as datetime_parser


@dataclass(frozen=True)
class GlobalConfiguration:
    algorithm: str
    arity: str
    total_budget: list[int]
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

    result_output_folder_name: str

    pipe: bool

    T = TypeVar('T')

    @staticmethod
    def _construct_from_dict(args: dict[str, Any], key: str, required_type: Type[T], default: Optional[T] = None) -> T:
        if key not in args:
            if default is not None:
                return default
            raise KeyError(key)
        if not isinstance(args[key], required_type):
            if default is not None:
                return default
            raise TypeError(f'Required type for {key=}: {required_type}. Got type {type(args[key])}.')
        return args[key]


    @classmethod
    def default(cls, args: dict[str, Any]):
        algorithm = cls._construct_from_dict(args, 'algorithm', str)
        arity = cls._construct_from_dict(args, 'arity', str)
        now = cls._construct_from_dict(args, 'now', datetime.datetime)
        now_date = f'{now.year}{now.month:02d}{now.day:02d}'
        now_time = f'{now.hour}{now.minute:02d}{now.second:02d}'
        return cls(
            algorithm=algorithm,
            arity=arity,
            total_budget=[10000, 100000],
            sampling_methods=['smallest-value', 'longest-value', 'random', 'evenly-spaced', 'first', 'biggest-value'],
            header=cls._construct_from_dict(args, 'header', bool),
            clip_output=cls._construct_from_dict(args, 'clip_output', bool),
            print_inds=cls._construct_from_dict(args, 'print_inds', bool),
            create_plots=cls._construct_from_dict(args, 'create_plots', bool),
            now=now,
            source_dir=cls._construct_from_dict(args, 'source_dir', str),
            tmp_folder=cls._construct_from_dict(args, 'tmp_folder', str),
            results_folder=cls._construct_from_dict(args, 'results_folder', str),
            results_suffix=cls._construct_from_dict(args, 'results_suffix', str),
            output_folder=cls._construct_from_dict(args, 'output_folder', str),
            result_output_folder_name=cls._construct_from_dict(args, 'result_name', str, f'output_{arity}_{now_date}_{now_time}'),
            pipe=cls._construct_from_dict(args, 'pipe', bool)
        )
        
    @staticmethod
    def argparse_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Modifies the provided argparse parser by adding optional arguments for all config options"""
        parser.add_argument('--algorithm', type=str, required=False, default=['BINDER', 'PartialSPIDER'][1], help='Which algorithm to use, either PartialSPIDER or BINDER')
        parser.add_argument('--arity', type=str, required=False, default=['unary', 'nary'][0], help='Whether to find `unary` or `nary` INDs')
        parser.add_argument('--now', type=datetime_parser.parse, required=False, default=datetime.datetime.now(), help='The time to use as `now`')
        parser.add_argument('--header', action=BooleanOptionalAction, required=False, default=False, help='Whether the provided csv files have headers')
        parser.add_argument('--clip-output', action=BooleanOptionalAction, required=False, default=True, help='Whether to clip the Metanome CLI output to onyl show the runtime')
        parser.add_argument('--print-inds', action=BooleanOptionalAction, required=False, default=False, help='Whether to print the found INDs')
        parser.add_argument('--create-plots', action=BooleanOptionalAction, required=False, default=True, help='Whether to create plots')
        parser.add_argument('--source-dir', type=str, required=False, default='src/', help='The directory containing the data files to sample and experiment on')
        parser.add_argument('--tmp-folder', type=str, required=False, default='tmp/', help='The directory containing temporary files while the program runs')
        parser.add_argument('--results-folder', type=str, required=False, default='results/', help='The directory containing temporary result files while the program runs')
        parser.add_argument('--results-suffix', type=str, required=False, default='_inds', help='What to append to the result files')
        parser.add_argument('--output-folder', type=str, required=False, default='output/', help='The directory containing result files after the program terminates')
        parser.add_argument('--result-name', type=str, required=False, default=None, help='The name of the run. Used to generate the output folder. Depends on the current time by default.')
        parser.add_argument('--pipe', action=BooleanOptionalAction, required=False, default=False, help='Whether to allow piping the output directly to the evaluation script')
        return parser
