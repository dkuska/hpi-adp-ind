from dataclasses import dataclass
from dataclasses_json import dataclass_json

from pysrc.utils.dataclass_json import DataclassJson


@dataclass_json
@dataclass(frozen=True)
class ColumnInformation(DataclassJson):
    """Contains information about a column"""
    table_name: str
    column_name: str

    def __repr__(self) -> str:
        return f'{self.table_name}.{self.column_name}'
