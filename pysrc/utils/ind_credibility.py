from pysrc.models.ind import IND
from ..models import metanome_run
import numpy as np

def ind_credibility(ind: IND, run: 'metanome_run.MetanomeRun', missing_values: int, baseline: 'metanome_run.MetanomeRun', model) -> float:
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
    
    
    #     ['missing_values', 'left_baseline_count', 'left_baseline_unique_ratio',
    #    'right_baseline_count', 'right_baseline_unique_ratio', 'left_count',
    #    'left_sampling_rate', 'right_count', 'right_sampling_rate',
    #    'cardinality_ratio', 'sample_size_ratio', 'missing_ratio',
    #    'useless_ratio']
    
    # TODO: Add sampling ratio to dependent stats
    stats = np.array([missing_values, 
                      baseline_dependents_stat.count, baseline_dependents_stat.unique_ratio, 
                      baseline_referenced_stat.count, baseline_dependents_stat.unique_ratio,
                      dependents_stat.count, dependents_stat.count / baseline_dependents_stat.count, 
                      referenced_stat.count, referenced_stat.count / baseline_referenced_stat.count,
                      ratio_of_cardinality, ratio_of_sample_sizes,
                      missing_ratio, useless_ratio]).reshape(1, -1)
    score = model.predict_proba(stats)
    return score[-1] # We are only interested in the probability that the IND is a TP
    
    
    # return (1.0 - missing_values / dependents_stat.unique_count) * run.configuration.credibility()
