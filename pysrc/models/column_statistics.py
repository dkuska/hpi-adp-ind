from dataclasses import dataclass

@dataclass
class ColumnStatistic():
    count: int
    unique_count: int
    unique_ratio: float
    min: str
    max: str
    shortest: str
    longest: str