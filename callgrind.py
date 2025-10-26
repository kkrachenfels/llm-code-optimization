import json
import subprocess
from collections import OrderedDict

def run_under_callgrind(executable_path, output_file, args=[]):
    """
    Runs the given executable under Callgrind and generates a Callgrind output file.
    
    Parameters:
    - executable_path (str): The path to the executable to be profiled.
    - output_file (str): The path where the Callgrind output file will be saved.
    - args (list): Additional arguments to pass to the executable.
    """
    command = ['valgrind', '--tool=callgrind', f'--callgrind-out-file={output_file}', executable_path] + args
    subprocess.run(command)


def annotate_callgrind_output(callgrind_file, annotated_file):
    """
    Annotates the Callgrind output file with source code information.
    
    Parameters:
    - callgrind_file (str): The path to the Callgrind output file.
    - source_file (str): The path to the source file to be annotated.
    """
    with open(annotated_file, 'w', encoding='utf-8') as outfile:
        subprocess.run([
            'callgrind_annotate',
            '--auto=yes',
            '--tree=both',
            '--inclusive=yes',
            callgrind_file
        ], stdout=outfile)


def replace_in_file(file_path, old_string, new_string):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    updated_content = content.replace(old_string, new_string)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)


def clean_callgrind_output(annotated_file):
    """
    Cleans the annotated Callgrind output file by removing blocks where the function line contains "dyld" (case insensitive).
    Each block is separated by blank lines and starts with a line containing ' * ' identifying the function.

    Parameters:
    - annotated_file (str): The path to the annotated Callgrind output file.
    """
    with open(annotated_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    cleaned_lines = []
    current_block = []
    in_block = False
    remove_block = False

    for line in lines:
        line = line.replace("???:", "") # Remove '???:' from lines
        # Detect start of a block: a line containing ' * '
        if ' * ' in line:
            # If we were in a previous block, decide whether to keep it
            if in_block and not remove_block:
                cleaned_lines.extend(current_block)
                cleaned_lines.append('\n')  # preserve the blank line separating blocks
            # Start a new block
            current_block = [line]
            in_block = True
            # Check if this function line contains "dyld" (case insensitive)
            remove_block = ('dyld' in line.lower() or 'dylib' in line.lower())
        elif line.strip() == '':
            # Blank line indicates end of current block
            if in_block:
                if not remove_block:
                    cleaned_lines.extend(current_block)
                    cleaned_lines.append('\n')  # preserve blank line
                current_block = []
                in_block = False
                remove_block = False
            else:
                # Outside block, just keep blank lines
                cleaned_lines.append(line)
        else:
            # Inside a block, accumulate lines
            if in_block:
                current_block.append(line)
            else:
                # Outside any block, just keep lines
                cleaned_lines.append(line)

    # Handle last block if file does not end with blank line
    if in_block and not remove_block:
        cleaned_lines.extend(current_block)

    with open(annotated_file, 'w', encoding='utf-8') as file:
        file.writelines(cleaned_lines)



def get_top_n_hotspots(annotated_file, top_n = 5):

    with open(annotated_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    fn_callstacks_by_percent = OrderedDict()
    cur_fn_lines = []
    cur_fn_info = {}

    for line in lines:
        line = line.strip()
        if line == '\n' or line == '':
            if len(cur_fn_lines) > 0 and len(cur_fn_info) > 0:
                fn_callstacks_by_percent[cur_fn_info['name']] = {
                    'info': cur_fn_info,
                    'callstack': cur_fn_lines
                }
            cur_fn_lines = []
            cur_fn_info = {}
            continue

        if ' * ' in line or ' > ' in line or ' < ' in line:
            # cur_fn_lines.append(line)
            num_calls = int(line.split('(')[0].strip().replace(',', ''))
            percent_calls = float(line.split('(')[1].split(')')[0].strip().replace('%', ''))
            fn_type = line.split(')')[1].strip()[0] # '*' or '>' or '<'
            fn_name = line.split(f' {fn_type} ')[1].strip()
            fn_info = {
                'name': fn_name,
                'num_calls': num_calls,
                'percent_calls': percent_calls,
                'type': fn_type
            }
            if fn_type == '*':
                cur_fn_info = fn_info
            cur_fn_lines.append(fn_info)

    if len(cur_fn_lines) > 0 and len(cur_fn_info) > 0:
        fn_callstacks_by_percent[cur_fn_info['name']] = {
            'info': cur_fn_info,
            'callstack': cur_fn_lines
        }

    top_n_hotspots = []
    for fn_name, info_and_callstack in fn_callstacks_by_percent.items():
        # don't count the main fn as a hotspot
        if 'main' in fn_name:
            continue
        
        if len(top_n_hotspots) >= top_n:
            break
        top_n_hotspots.append(info_and_callstack)

    # print(json.dumps(top_n_hotspots, indent=2))
    return top_n_hotspots


def test():
    # testing w the quant dataset
    project_dir = 'datasets/quantpp'
    executable_path = f'{project_dir}/quant'
    executable_args = [f'{project_dir}/data/table.csv']
    callgrind_output = f'{project_dir}/callgrind.out.test'
    annotated_output = f'{project_dir}/annotated_callgrind.out'

    # run_under_callgrind(executable_path, callgrind_output, executable_args)
    annotate_callgrind_output(callgrind_output, annotated_output)
    clean_callgrind_output(annotated_output)
    hotspots = get_top_n_hotspots(annotated_output, 1)
    print("Identified Hotspots:")
    for hotspot in hotspots:
        print(hotspot['info'])


if __name__ == "__main__":
    test()