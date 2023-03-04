from dataclasses import astuple, dataclass
from pysrc.models.errors import INDType
from pysrc.models.ind import IND
from pysrc.models.metanome_run import MetanomeRun
from pysrc.models.metanome_run_results import MetanomeRunResults


@dataclass(frozen=True)
class LineComparisonResultUnary:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_tp_missing_values: float
    mean_fp_missing_values: float
    
    def unpack(self) -> tuple[int, int, int, float, float, float, float, float]:
        return self.tp, self.fp, self.fn, self.precision, self.recall, self.f1, self.mean_tp_missing_values, self.mean_fp_missing_values


@dataclass(frozen=True)
class LineComparisonResultNary:
    tp: list[int]
    fp: list[int]
    fn: list[int]
    precision: list[float]
    recall: list[float]
    f1: list[float]
    
    def unpack(self) -> tuple[list[int], list[int], list[int], list[float], list[float], list[float]]:
        return self.tp, self.fp, self.fn, self.precision, self.recall, self.f1


def compare_csv_line_unary(inds: list[IND], baseline: MetanomeRunResults) -> LineComparisonResultUnary:
    """For unary INDs, this method returns absolute counts for TP, FP, FN, etc."""
    tp, fp = 0, 0
    sum_tp_missing_values, sum_fp_missing_values = 0, 0
    num_inds = len(inds)

    for ind in inds:
        if baseline.has_ind(ind):
            ind.errors.append(INDType('TP'))
            tp += 1
            sum_tp_missing_values += ind.missing_values('dict')
        else:
            ind.errors.append(INDType('FP'))
            fp += 1
            sum_fp_missing_values += ind.missing_values('dict')

    fn = len(baseline.inds) - tp

    nan = float('nan')
    if num_inds > 0:
        precision = tp / (tp + fp) if tp + fp != 0 else nan
        recall = tp / (tp + fn) if tp + fn != 0 else nan
        f1 = 2 * (precision * recall) / (precision + recall) if recall + precision != 0 else nan
        mean_tp_missing_values = sum_tp_missing_values / tp if tp > 0 else nan
        mean_fp_missing_values = sum_fp_missing_values / fp if fp > 0 else nan
    else:
        precision, recall, f1, mean_tp_missing_values, mean_fp_missing_values = 0.0, 0.0, 0.0, 0.0, 0.0

    return LineComparisonResultUnary(tp, fp, fn, precision, recall, f1, mean_tp_missing_values, mean_fp_missing_values)


def compare_csv_line_nary(inds: list[IND], baseline: MetanomeRunResults) -> LineComparisonResultNary:
    """For nary INDs, this returns lists with counts for each arity"""
    max_arity = max([ind.arity() for ind in baseline.inds])

    tp, fp = [0 for _ in range(max_arity)], [0 for _ in range(max_arity)]
    inds_per_arity = [0 for _ in range(max_arity)]
    for ind in baseline.inds:
        inds_per_arity[ind.arity() - 1] += 1

    for ind in inds:
        arity = ind.arity() - 1  # -1 to match list indices
        if baseline.has_ind(ind):
            ind.errors.append(INDType('TP'))
            tp[arity] += 1
        else:
            ind.errors.append(INDType('FP'))
            fp[arity] += 1

    fn = [inds_per_arity[arity] - tp[arity] for arity in range(max_arity)]

    precision, recall, f1 = [0.0 for _ in range(max_arity)], [0.0 for _ in range(max_arity)], [0.0 for _ in
                                                                                               range(max_arity)]
    for i in range(max_arity):
        if tp[i] + fp[i] > 0:
            precision[i] = tp[i] / (tp[i] + fp[i])

        if tp[i] + fn[i] > 0:
            recall[i] = tp[i] / (tp[i] + fn[i])

        if recall[i] + precision[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = float('nan')

    return LineComparisonResultNary(tp, fp, fn, precision, recall, f1)


def run_as_compared_csv_line(run: MetanomeRun, baseline: MetanomeRunResults) -> list[str]:
    sampled_file_paths = run.configuration.source_files
    sampled_file_names = [path.rsplit('/', 1)[-1].replace('.csv', '') for path in sampled_file_paths]

    file_names, methods, budgets = [], [], []
    for sampled_file in sampled_file_names:
        split_filename = sampled_file.split(
            '__')  # Detect Column Sampling, as this is evident from the '__' in the file
        split_metadata = []
        if len(split_filename) == 2:
            split_metadata = split_filename[1].split('_')
        split_filename = [split_filename[0]]

        if len(split_metadata) == 3:
            split_filename.append(split_metadata[0])
            split_filename.append(split_metadata[1])
        if len(split_filename) == 3:
            fname, budget, sampling_method = split_filename
            fname = fname + '_' + split_metadata[2]
        else:
            fname, budget, sampling_method = sampled_file, str(float('inf')), 'None'

        file_names.append(fname)
        methods.append(sampling_method)
        budgets.append(budget)

    if run.configuration.arity == 'unary':
        tp, fp, fn, precision, recall, f1, mean_tp_missing_values, mean_fp_missing_values = compare_csv_line_unary(run.results.inds, baseline).unpack()
        return ['; '.join(file_names), '; '.join(methods), '; '.join(budgets), str(tp), str(fp), str(fn), f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}', f'{mean_tp_missing_values:.3f}', f'{mean_fp_missing_values:.3f}']

    else:
        tp, fp, fn, precision, recall, f1 = compare_csv_line_nary(run.results.inds, baseline).unpack()

        return ['; '.join(file_names),
                '; '.join(methods),\
                '; '.join(budgets), \
                '; '.join([str(tp_i) for tp_i in tp]), \
                '; '.join([str(fp_i) for fp_i in fp]), \
                '; '.join([str(fn_i) for fn_i in fn]), \
                '; '.join([f'{precision_i:.3f}' for precision_i in precision]), \
                '; '.join([f'{recall_i:.3f}' for recall_i in recall]), \
                '; '.join([f'{f1_i:.3f}' for f1_i in f1])]
