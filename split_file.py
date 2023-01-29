import os
import sys


def main() -> None:
    file_name = sys.argv[1]
    pipe = '--pipe' in sys.argv
    subfiles: list[list[str]] = []
    current_subfile_index = -1
    with open(file_name, 'r') as f:
        for current_line in f:
            if '*' * 50 == current_line.rstrip():
                current_subfile_index += 1
                subfiles.append([])
                continue
            subfiles[current_subfile_index].append(current_line)
    # subfile_names: list[str] = []
    for subfile in subfiles:
        title = subfile[0].rsplit(os.sep, 3)[1]
        output_file = f'{file_name}-{title}'
        # subfile_names.append(output_file)
        with open(output_file, 'w+') as f:
            f.writelines(subfile)
        if pipe:
            print(f'{output_file}\0', end='', flush=True)
        with open(f'{output_file}-condensed', 'w+') as f:
            f.writelines(line for line in subfile if 'ranked_ind.credibility' not in line)
    # for subfile_name in subfile_names:
    #     subsections: list[tuple[str, list[str]]] = []
    #     current_subsection_index = -1
    #     with open(subfile_name, 'r') as f:
    #         for current_line in f:
    #             if 'Results for file=' in current_line or 'Error metrics collection is (temporarily) disabled for performance reasons' in current_line:
    #                 continue
    #             if 'allowed_baseline_knowledge' in current_line:
    #                 current_subsection_index += 1
    #                 mode = current_line.split('\'')[1]
    #                 subsections.append((mode, []))
    #                 continue
    #             try:
    #                 subsections[current_subsection_index][1].append(current_line)
    #             except Exception as ex:
    #                 print(f'{current_subsection_index=}, {subsections=}, {current_line=}, {subfile_name=}')
    #                 raise ex
    #     for subsection in subsections:
    #         output_file = f'{subfile_name}-{subsection[0]}'
    #         with open(output_file, 'w+') as f:
    #             f.writelines(subsection[1])

if __name__ == '__main__':
    main()
