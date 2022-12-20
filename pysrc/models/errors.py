from abc import ABC
from dataclasses import dataclass
from typing import Literal
from dataclasses_json import dataclass_json

class ErrorMetric(ABC):
    pass


@dataclass_json
@dataclass(frozen=True)
class TuplesToRemove(ErrorMetric):
    absolute_tuples_to_remove: int
    relative_tuples_to_remove: float
    absolute_distinct_tuples_to_remove: int
    relative_distinct_tuples_to_remove: float


@dataclass_json
@dataclass(frozen=True)
class INDType(ErrorMetric):
    ind_type: Literal['TP', 'FP']


@dataclass_json
@dataclass(frozen=True)
class MissingValues(ErrorMetric):
    missing_values: int
