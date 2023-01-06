from dataclasses import dataclass
from dataclasses_json import dataclass_json

from .column_information import ColumnInformation

@dataclass_json
@dataclass(frozen=True)
class ColumnStatistic:
    
    column_Information: ColumnInformation
    count: int
    unique_count: int
    unique_ratio: float
    min: str
    max: str
    shortest: str
    longest: str