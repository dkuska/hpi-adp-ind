from dataclasses import dataclass

from models.column_information import ColumnInformation


@dataclass(frozen=True)
class IND:
    """Represents a single, nnary ind"""
    dependants: list[ColumnInformation]

    referenced: list[ColumnInformation]

    def __repr__(self) -> str:
        return f'{" & ".join(self.dependants)} [= {" & ".join(self.referenced)}'
