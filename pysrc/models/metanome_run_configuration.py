from dataclasses import dataclass
import datetime
from dataclasses_json import dataclass_json

from pysrc.utils.dataclass_json import DataclassJson


@dataclass_json
@dataclass(frozen=True)
class MetanomeRunConfiguration(DataclassJson):
    """Contains configuration information about a Metanome run"""
    algorithm: str
    arity: str
    total_budget: int | None
    sampling_method: str
    allowed_missing_values: int
    time: datetime.datetime

    source_dir: str
    source_files: list[str]
    tmp_folder: str
    results_folder: str
    result_suffix: str
    output_folder: str
    clip_output: bool
    header: bool
    print_inds: bool
    create_plots: bool

    is_baseline: bool

    def __hash__(self) -> int:
        return hash((self.algorithm, self.arity, self.total_budget, self.sampling_method, self.allowed_missing_values, self.time, self.source_dir,
                     tuple(self.source_files), self.tmp_folder, self.results_folder,
                     self.result_suffix, self.output_folder, self.clip_output, self.header,
                     self.print_inds, self.create_plots, self.is_baseline))

    def credibility(self) -> float:
        """Get the credibility (i.e. how trustworthy this config is) of the config."""
        # TODO: Actually depend this on the config data
        # return float(product([budget for budget in self.total_budget]))
        if self.total_budget is None:
            raise ValueError(self.total_budget)
        return self.total_budget
