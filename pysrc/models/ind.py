from dataclasses import dataclass

from pysrc.models.column_information import ColumnInformation


@dataclass(frozen=True)
class IND:
    """Represents a single, nary ind"""
    dependents: list[ColumnInformation]

    referenced: list[ColumnInformation]

    def __repr__(self) -> str:
        return f'{" & ".join(str(d) for d in self.dependents)} [= {" & ".join(str(r) for r in self.referenced)}'

    def __hash__(self) -> int:
        return hash((tuple(self.dependents), tuple(self.referenced)))
    
    def arity(self) -> int:
        return 1 if len(self.dependents) == 1 else len(self.dependents)