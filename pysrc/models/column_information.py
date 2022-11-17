from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnInformation:
    """Contains information about a column"""
    table_name: str
    column_name: str

    def __repr__(self) -> str:
        return f'{self.table_name}.{self.column_name}'
