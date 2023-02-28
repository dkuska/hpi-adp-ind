from dataclasses import dataclass
from dataclasses_json import dataclass_json

from pysrc.utils.dataclass_json import DataclassJson

from .column_information import ColumnInformation

@dataclass_json
@dataclass(frozen=True)
class ColumnStatistic(DataclassJson):
    column_information: ColumnInformation
    count: int
    unique_count: int
    unique_ratio: float
    min: str
    max: str
    shortest: str
    longest: str
