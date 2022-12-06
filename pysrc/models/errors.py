from abc import ABC
from dataclasses import dataclass
from typing import Literal


class ErrorMetric(ABC):
    pass


@dataclass(frozen=True)
class TuplesToRemove(ErrorMetric):
    absolute_tuples_to_remove: int
    relative_tuples_to_remove: float
    absolute_distinct_tuples_to_remove: int
    relative_distinct_tuples_to_remove: float


@dataclass(frozen=True)
class INDType(ErrorMetric):
    ind_type: Literal['TP', 'FP']

@dataclass(frozen=True)
class MissingValues(ErrorMetric):
    missing_values: int