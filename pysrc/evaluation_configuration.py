from argparse import ArgumentParser
import argparse
from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar


@dataclass(frozen=True)
class EvaluationConfiguration:
    file: Optional[str]
    return_path: str
    interactive: bool
    top_inds: int

    T = TypeVar('T')

    @staticmethod
    def _construct_from_dict(args: dict[str, Any], key: str, required_type: Type[T],
                             may_be_none: bool = False, default: Optional[T] = None) -> T:
        if key not in args:
            if default is not None or may_be_none:
                return default
            raise KeyError(key)
        if not isinstance(args[key], required_type):
            if default is not None:
                return default
            raise TypeError(f'Required type for {key=}: {required_type}. Got type {type(args[key])}.')
        return args[key]

    @classmethod
    def default(cls, args: dict[str, Any]) -> 'EvaluationConfiguration':
        file = cls._construct_from_dict(args, 'file', Optional[str], True)
        return_path = cls._construct_from_dict(args, 'return_path', str)
        interactive = cls._construct_from_dict(args, 'interactive', bool)
        top_inds = cls._construct_from_dict(args, 'top_inds', int)
        return cls(
            file=file,
            return_path=return_path,
            interactive=interactive,
            top_inds=top_inds
        )
        
    @staticmethod
    def argparse_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Modifies the provided argparse parser by adding optional arguments for evaluation config options"""
        parser.add_argument('--file', type=str, required=False, default=None,
                            help='The JSON file containing the experiment information to be evaluated. Mutually exclusive to `--pipe`.')
        parser.add_argument('--return-path', type=str, required=False, default=None,
                            help='Whether to return no path (default), the path of the created csv file (`csv`), of the plot (`plot`), or of the ranked inds (`ranked`)')
        parser.add_argument('--interactive', action=argparse.BooleanOptionalAction, required=False, default=False,
                            help='Whether to print the error metrics in a human-readable way')
        parser.add_argument('--top-inds', type=int, default=-1,
                            help='The number of INDs (from the top ranking) that should be shown. A negative number shows all.')
        return parser
