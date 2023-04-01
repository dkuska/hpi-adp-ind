from dataclasses import dataclass
from dataclasses_json import dataclass_json

from pysrc.models.column_statistics import ColumnStatistic
from pysrc.models.metanome_run_configuration import MetanomeRunConfiguration
from pysrc.models.metanome_run_results import MetanomeRunResults
from pysrc.utils.dataclass_json import DataclassJson


@dataclass_json
@dataclass(frozen=True)
class MetanomeRun(DataclassJson):
    configuration: MetanomeRunConfiguration
    column_statistics: list[ColumnStatistic]
    results: MetanomeRunResults
