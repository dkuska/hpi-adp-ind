from abc import ABC
from dataclasses import dataclass
from typing import Literal
from dataclasses_json import dataclass_json

from pysrc.utils.dataclass_json import DataclassJson


@dataclass_json
class ErrorMetric(ABC, DataclassJson):
    pass


@dataclass_json
@dataclass(frozen=True)
class INDType(ErrorMetric):
    ind_type: Literal['TP', 'FP']


@dataclass_json
@dataclass(frozen=True)
class MissingValues(ErrorMetric):
    missing_values: int
