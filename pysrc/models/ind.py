from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from pysrc.models.column_information import ColumnInformation
from pysrc.models.errors import ErrorMetric


@dataclass_json
@dataclass(frozen=True)
class IND:
    """Represents a single, nary ind"""
    dependents: list[ColumnInformation]

    referenced: list[ColumnInformation]

    errors: list[ErrorMetric] = field(default_factory=list, compare=False) #NOTE: THIS IS JUST TEMPORARY SO THAT WE CAN ACTUALLY WORK WITH PARTIAL SPIDER!!!

    def __repr__(self) -> str:
        return f'{" & ".join(str(d) for d in self.dependents)} [= {" & ".join(str(r) for r in self.referenced)}'

    def __hash__(self) -> int:
        return hash((tuple(self.dependents), tuple(self.referenced)))
    
    
    def arity(self) -> int:
        return len(self.dependents)