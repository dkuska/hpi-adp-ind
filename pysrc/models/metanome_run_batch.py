from dataclasses import dataclass
from math import isnan
from typing import Iterator
from dataclasses_json import dataclass_json
from pysrc.models.ind import IND
from pysrc.models.metanome_run import MetanomeRun
from pysrc.utils.dataclass_json import DataclassJson
from pysrc.utils.ind_credibility import ind_credibility


@dataclass_json
@dataclass(frozen=True)
class MetanomeRunBatch(DataclassJson):
    runs: list[MetanomeRun]

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator[MetanomeRun]:
        return self.runs.__iter__()

    @property
    def baseline(self) -> MetanomeRun:
        return next(run for run in self.runs if run.configuration.is_baseline)

    def ranked_inds(self) -> dict[IND, float]:
        # Collect INDs
        ind_map: dict[tuple[str, str], IND] = {}  # Map from (dependent, referenced) -> IND
        inds: dict[IND, list[tuple[int, MetanomeRun]]] = {}
        for run in self.runs:
            for ind in run.results.inds:
                if (str(ind.dependents), str(ind.referenced)) not in ind_map:
                    clean_ind = IND(dependents=ind.dependents, referenced=ind.referenced)
                    ind_map[(str(ind.dependents), str(ind.referenced))] = clean_ind
                    inds[clean_ind] = []

        for run in self.runs:
            # Skip baseline (it won't be available in production)
            # TODO: Is this correct behavior? Not sure, but otherwise we'll get *every* TP IND as we're checking the baseline.
            if run.configuration.is_baseline:
                continue
            for ind in run.results.inds:
                clean_ind = ind_map[(str(ind.dependents), str(ind.referenced))]
                # if clean_ind not in inds: inds[clean_ind] = []
                inds[clean_ind].append((ind.missing_values('dict'), run))
        # maximum_missing_values = max(missing_values for configMissingValuePairs in inds.values() for missing_values, _ in configMissingValuePairs)
        baseline = self.baseline
        inds_credibilities = {
                ind: [
                    ind_credibility(ind, run, missing_values, baseline)
                    for missing_values, run
                    in configMissingValuesPairs
                    ]
                for ind, configMissingValuesPairs
                in inds.items()
            }
        for ind in inds_credibilities:
            if len(inds_credibilities[ind]) > 0:
                continue
            # When this IND was not found in the runs, set -2.
            inds_credibilities[ind] = [-2.0]
        # Rank INDs by SUM over ALL runs (with value 0.0 for runs that didn't find the IND)
        ranked_inds = { ind: credibility_sum if not isnan(credibility_sum := sum(credibilities)) else -1.0 for ind, credibilities in inds_credibilities.items() }
        return ranked_inds
