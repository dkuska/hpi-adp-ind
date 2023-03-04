from dataclasses import dataclass, field
from typing import Literal, Optional
from dataclasses_json import dataclass_json

from pysrc.models.column_information import ColumnInformation
from pysrc.models.errors import ErrorMetric, MissingValues
from pysrc.utils.dataclass_json import DataclassJson


@dataclass_json
@dataclass(frozen=True)
class IND(DataclassJson):
    """Represents a single, nary ind.
    For unary INDs, exactly one element is present in `dependents` and `referenced`."""
    dependents: list[ColumnInformation]

    referenced: list[ColumnInformation]

    errors: list[ErrorMetric] = field(default_factory=list, compare=False)

    def __repr__(self) -> str:
        return f'{" & ".join(str(d) for d in self.dependents)} [= {" & ".join(str(r) for r in self.referenced)}'

    def __hash__(self) -> int:
        return hash((tuple(self.dependents), tuple(self.referenced)))
    
    def arity(self) -> int:
        return len(self.dependents)

    # TODO: Remove 'dict' method when possible
    # Requires proper deserialization in the evaluation phase
    def missing_values(self, mode: Literal['object', 'dict']) -> int:
        # If it doesn't exist we're the baseline, so insert "fake" 0
        if mode == 'dict':
            missing_values_error: MissingValues = next((MissingValues(error['missing_values']) for error in self.errors if isinstance(error, dict) and 'missing_values' in error), MissingValues(0))
        elif mode == 'object':
            missing_values_error = next((error for error in self.errors if isinstance(error, MissingValues)), MissingValues(0))
        return missing_values_error.missing_values


@dataclass_json
@dataclass(frozen=True)
class RankedIND(DataclassJson):
    ind: IND
    credibility: float
    is_tp: Optional[bool]
