from abc import ABC
from dataclasses import dataclass


class ErrorMetric(ABC):
    pass


@dataclass(frozen=True)
class TuplesToRemove(ErrorMetric):
    absolute_tuples_to_remove: int
    relative_tuples_to_remove: float
