from typing import Literal
from pysrc.models.ind import IND
from pysrc.utils.eprint import eprint
from ..models import metanome_run


def ind_credibility(ind: IND, run: 'metanome_run.MetanomeRun', missing_values: int, baseline: 'metanome_run.MetanomeRun', allowed_baseline_knowledge: Literal['all', 'count', 'none']) -> float:
    dependents_stats = [next(stat for stat in run.column_statistics if stat.column_information == dependent) for dependent in ind.dependents]
    referenced_stats = [next(stat for stat in run.column_statistics if stat.column_information == referenced) for referenced in ind.referenced]
    baseline_dependents_stats = [next(stat for stat in baseline.column_statistics if stat.column_information == dependent) for dependent in ind.dependents]
    baseline_referenced_stats = [next(stat for stat in baseline.column_statistics if stat.column_information == referenced) for referenced in ind.referenced]
    # Only consider unary INDs
    dependents_stat = dependents_stats[0]
    referenced_stat = referenced_stats[0]
    baseline_dependents_stat = baseline_dependents_stats[0]
    baseline_referenced_stat = baseline_referenced_stats[0]
    # # Calculate (possibly) relevant stats
    # sampling_ratio_dependent = baseline_dependents_stat.unique_count / dependents_stat.unique_count
    # sampling_ratio_referenced = baseline_referenced_stat.unique_count / referenced_stat.unique_count
    # ratio_of_sample_sizes = dependents_stat.unique_count / referenced_stat.unique_count
    # ratio_of_cardinality = baseline_dependents_stat.unique_count / baseline_referenced_stat.unique_count
    # missing_ratio = missing_values / dependents_stat.unique_count
    # useless_ratio = missing_values / referenced_stat.unique_count
    # Check plausibility
    nan = float('nan')
    if baseline_dependents_stat.unique_count > baseline_referenced_stat.unique_count:
        # There are more dependent values than referenced ones
        # if 'ANALYSIS.column12 [= ANALYSIS.column5' in ind.__repr__() or 'ANALYSIS.column12 [= ANALYSIS.column8'  in ind.__repr__():
        #     eprint(f'1: {ind=} {baseline_dependents_stat.unique_count=} {baseline_referenced_stat.unique_count=} {ind.errors=} {run=}')
        #     exit(121)
        if allowed_baseline_knowledge in ['all', 'count']:
            return nan
    if baseline_dependents_stat.min < baseline_referenced_stat.min or baseline_dependents_stat.max > baseline_referenced_stat.max:
        # Minimum of dependent is smaller than minimum of referenced (or analogue with maximum)
        # if 'ANALYSIS.column12 [= ANALYSIS.column5' in ind.__repr__() or 'ANALYSIS.column12 [= ANALYSIS.column8'  in ind.__repr__():
        #     eprint(f'2: {ind=} {baseline_dependents_stat.min=} {baseline_referenced_stat.min=} {baseline_dependents_stat.max=} {baseline_referenced_stat.max=} {ind.errors=} {run=}')
        #     exit(122)
        if allowed_baseline_knowledge in ['all']:
            return nan
    if missing_values > baseline_referenced_stat.unique_count - referenced_stat.unique_count:
        # There are more missing values than there are values that are not in the sample of the right hand side
        # if 'ANALYSIS.column12 [= ANALYSIS.column5' in ind.__repr__() or 'ANALYSIS.column12 [= ANALYSIS.column8'  in ind.__repr__():
        #     eprint(f'3: {ind=} {missing_values=} {baseline_referenced_stat.unique_count=} {referenced_stat.unique_count=} {ind.errors=} {run=}')
        #     exit(123)
        if allowed_baseline_knowledge in ['all', 'count']:
            return nan
    cred = (1.0 - missing_values / dependents_stat.unique_count) * run.configuration.credibility()
    if cred < 0:
        eprint(f'Got negative credibility ({cred=}) for {ind=} with {ind.errors=}.\n\t{missing_values=}\n\t{dependents_stat.unique_count=}\n\t{run=}')
        # eprint(f'{missing_values=}, {dependents_stat.unique_count=}, {run=}, {ind=}')
        # exit(1)
    return cred
