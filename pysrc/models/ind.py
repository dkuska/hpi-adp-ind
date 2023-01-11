from dataclasses import dataclass, field
from typing import Optional
from dataclasses_json import dataclass_json

from pysrc.models.column_information import ColumnInformation
from pysrc.models.errors import ErrorMetric
from pysrc.models.metanome_run import MetanomeRun


@dataclass_json
@dataclass(frozen=True)
class IND:
    """Represents a single, nary ind"""
    dependents: list[ColumnInformation]

    referenced: list[ColumnInformation]

    errors: list[ErrorMetric] = field(default_factory=list, compare=False)

    def __repr__(self) -> str:
        return f'{" & ".join(str(d) for d in self.dependents)} [= {" & ".join(str(r) for r in self.referenced)}'

    def __hash__(self) -> int:
        return hash((tuple(self.dependents), tuple(self.referenced)))
    
    
    def arity(self) -> int:
        return len(self.dependents)


@dataclass_json
@dataclass(frozen=True)
class RankedIND:
    ind: IND
    credibility: float
    is_tp: Optional[bool]


def ind_credibility(ind: IND, run: MetanomeRun, missing_values: int, baseline: MetanomeRun) -> float:
    dependents_stats = [next(stat for stat in run.column_statistics if stat.column_information == dependent) for dependent in ind.dependents]
    referenced_stats = [next(stat for stat in run.column_statistics if stat.column_information == referenced) for referenced in ind.referenced]
    baseline_dependents_stats = [next(stat for stat in baseline.column_statistics if stat.column_information == dependent) for dependent in ind.dependents]
    baseline_referenced_stats = [next(stat for stat in baseline.column_statistics if stat.column_information == referenced) for referenced in ind.referenced]
    # Only consider unary INDs
    dependents_stat = dependents_stats[0]
    referenced_stat = referenced_stats[0]
    baseline_dependents_stat = baseline_dependents_stats[0]
    baseline_referenced_stat = baseline_referenced_stats[0]
    # Calculate (possibly) relevant stats
    sampling_ratio_dependent = baseline_dependents_stat.unique_count / dependents_stat.unique_count
    sampling_ratio_referenced = baseline_referenced_stat.unique_count / referenced_stat.unique_count
    ratio_of_sample_sizes = dependents_stat.unique_count / referenced_stat.unique_count
    ratio_of_cardinality = baseline_dependents_stat.unique_count / baseline_referenced_stat.unique_count
    missing_ratio = missing_values / dependents_stat.unique_count
    useless_ratio = missing_values / referenced_stat.unique_count
    # Check plausibility
    nan = float('nan')
    if baseline_dependents_stat.unique_count > baseline_referenced_stat.unique_count:
        # There are more dependent values than referenced ones
        return nan
    if baseline_dependents_stat.min < baseline_referenced_stat.min or baseline_dependents_stat.max > baseline_referenced_stat.max:
        # Minimum of dependent is smaller than minimum of referenced (or analogue with maximum)
        return nan
    if missing_values > baseline_referenced_stat.unique_count - referenced_stat.unique_count:
        # There are more missing values than there are values that are not in the sample of the right hand side
        return nan
    return (1.0 - missing_values / dependents_stat.unique_count) * run.configuration.credibility()
