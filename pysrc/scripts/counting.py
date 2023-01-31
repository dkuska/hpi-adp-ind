import sys

from pysrc.scripts.evaluation import load_experiment_information


def main() -> None:
    file_name = sys.argv[1]
    batch = load_experiment_information(file_name)
    print('Done loading...', flush=True)
    baseline = batch.baseline
    count = 0
    distinct_count = 0
    for stat in baseline.column_statistics:
        count += stat.count
        distinct_count += stat.unique_count
    attribute_count = len(baseline.column_statistics)
    ind_count = len(baseline.results.inds)
    print(f'{file_name= }')
    print(f'{count= }')
    print(f'{distinct_count= }')
    print(f'{attribute_count= }')
    print(f'{ind_count= }')
    


if __name__ == '__main__':
    main()
