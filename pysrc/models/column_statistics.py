from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass(frozen=True)
class ColumnStatistic:
    count: int
    unique_count: int
    unique_ratio: float
    min: str
    max: str
    shortest: str
    longest: str