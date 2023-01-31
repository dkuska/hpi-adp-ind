from dataclasses_json import dataclass_json
from dataclasses import dataclass
import sys
from matplotlib import axes
from matplotlib import gridspec
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CHUNK_SIZE = 4096


def each_chunk(stream, separator):
  buffer = ''
  while True:  # until EOF
    chunk = stream.read(CHUNK_SIZE)  # I propose 4096 or so
    if not chunk:  # EOF?
      yield buffer
      break
    buffer += chunk
    while True:  # until no separator is found
      try:
        part, buffer = buffer.split(separator, 1)
      except ValueError:
        break
      else:
        yield part


# class EnhancedJSONDecoder(json.JSONDecoder):
#         def default(self, o):
#             if dataclasses.is_dataclass(o):
#                 return dataclasses.asdict(o)
#             return super().default(o)


@dataclass_json
@dataclass
class RankedIND:
    credibility: float
    is_tp: bool
    ind: str


@dataclass_json
@dataclass
class Data:
    file: str
    allowed_baseline_knowledge: str
    sampling_method: str
    budget: int
    inds: list[RankedIND]
    
    
def create_plot(data: Data, axes: axes.Axes) -> axes.Axes:
    data_dict: dict[str, list] = {
        'kind': [],
        'credibility': []
    }
    for ind in data.inds:
        data_dict['kind'].append('TP' if ind.is_tp else 'FP')
        data_dict['credibility'].append(ind.credibility)
    frame = pd.DataFrame(data=data_dict)
    return sns.violinplot(frame, x='kind', y='credibility', ax=axes, scale='width')


def main() -> None:
    current_file_name: str = ''
    current_sampling_method: str = ''
    current_budget: int = 0
    f: Figure | None = None
    axiis: list[list[axes.Axes]]
    for json_data in each_chunk(sys.stdin, '\0'):
        if json_data == '':
            continue
        data: Data = Data.from_json(json_data)
        # Determine axes
        if current_file_name != data.file or current_sampling_method != data.sampling_method or current_budget != data.budget:
            if f:
                plot_path = f'{current_file_name}_{current_sampling_method}_{current_budget}_plots.jpg'
                f.savefig(plot_path)
                print(plot_path)
                plt.close(f)
            current_file_name = data.file
            current_sampling_method = data.sampling_method
            current_budget = data.budget
            f, axiis = plt.subplots(1, 3, figsize=(15, 10))
            f.suptitle(f'Dataset: {current_file_name.rsplit("-", 2)[-1]}. Sampling Method: {current_sampling_method}. Budget: {data.budget}. Normalized.', fontsize=16)
        ax = axiis[
            0 if data.allowed_baseline_knowledge == 'all'
            else 1 if data.allowed_baseline_knowledge == 'count'
            else 2
            ]
        # Normalize data
        # Budget is upper bound for credibilities
        max_cred = data.budget  # max(ind.credibility for ind in data.inds)
        if max_cred > 0.0:
            for ind in data.inds:
                if ind.credibility < 0.0:
                    continue
                ind.credibility = ind.credibility / max_cred
        # Remove -2.0 (FN) INDs
        fn_count = sum(1 for ind in data.inds if ind.credibility < -1.5)
        data.inds = [ind for ind in data.inds if ind.credibility > -1.5]
        plot = create_plot(data, ax)
        plot.set_title(f'Baseline knowledge: {data.allowed_baseline_knowledge}. FNs: {fn_count}.')


# def get_index(orig: int, /) -> tuple[int, int]:
#     row = orig // 3
#     col = orig % 3
#     return (row, col)


# def main() -> None:
#     f: Figure
#     # axiis: list[list[axes.Axes]]
#     ax: axes.Axes
#     f = plt.figure()
#     ax = f.add_subplot(1, 1, 1)
#     initial = True
#     # f, axiis = plt.subplots(216, 3)
#     sns.despine(f)
#     # ax_index = 0
#     for json_data in each_chunk(sys.stdin, '\0'):  # .read().split('\0'):
#         if json_data == '':
#             continue
#         data: Data = Data.from_json(json_data)
#         # Normalize data
#         max_cred = max(ind.credibility for ind in data.inds)
#         if max_cred > 0.0:
#             for ind in data.inds:
#                 if ind.credibility < 0.0:
#                     continue
#                 ind.credibility = ind.credibility / max_cred
#         if not initial:
#             # Resize plots
#             n = len(f.axes) + 1
#             gs = gridspec.GridSpec(n, 1)
#             for i, ax in enumerate(f.axes):
#                 ax.set_position(gs[i].get_position(f))
#                 ax.set_subplotspec(gs[i])
#             ax = f.add_subplot(gs[n-1])
#             # n = len(f.axes)
#             # for i in range(n):
#             #     f.axes[i].change_geometry(n+1, 1, i+1)
#             # ax = f.add_subplot(n+1, 1, n+1)
#         else: initial = False
#         # plot = create_plot(data, axiis[get_index(ax_index)[0]][get_index(ax_index)[1]])
#         plot = create_plot(data, ax)
#         # ax_index += 1
#         plot.set_title(f'{data.sampling_method}, {data.budget}, {data.allowed_baseline_knowledge} (normalized)')

#     plot_path = f'{data.file}_plots.jpg'
#     f.savefig(plot_path)
#     print(plot_path)


if __name__ == '__main__':
    main()
