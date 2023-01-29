from dataclasses import dataclass
from dataclasses_json import dataclass_json
import sys


CHUNK_SIZE = 8  # Small because each input is small


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


def main() -> None:
    data: list[Data] = []
    data_index = -1
    for file in each_chunk(sys.stdin, '\0'):  # .read().split('\0'):
        if file == '':
            continue
        with open(file, 'r') as f:
            for line in f:
                if '*** Results' in line or 'Error metrics' in line:
                    continue
                if 'allowed_baseline_knowledge' in line:
                    if data_index >= 0:
                        # print(f'{json.dumps(data[data_index], cls=EnhancedJSONEncoder)}\0', end='', flush=True)
                        print(f'{data[data_index].to_json()}\0', end='', flush=True)
                    data.append(Data(file, 'None', 'None', -1, []))
                    data_index += 1
                    data[data_index].allowed_baseline_knowledge = line.split('\'')[1]
                    continue
                if 'Results for run.' in line:
                    method = line.split('\'', maxsplit=3)[1]
                    budget = int(line.rsplit(', ', maxsplit=3)[1])
                    data[data_index].budget = budget
                    data[data_index].sampling_method = method
                    continue
                if 'For maximum_threshold_percentage=' in line:
                    continue
                if 'ranked_ind.credibility=' in line:
                    cred = float(line.split('=', 2)[1].split(' ', 2)[0])
                    is_tp = line.split('=', 3)[2].split(')', 2)[0] == 'True'
                    ind = line.split('ind=')[1].rstrip()
                    ranked_ind = RankedIND(cred, is_tp, ind)
                    data[data_index].inds.append(ranked_ind)
                    continue
        print(f'{data[data_index].to_json()}\0', end='', flush=True)
        

if __name__ == '__main__':
    main()
