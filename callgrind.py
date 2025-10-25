import subprocess

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
            remove_block = 'dyld' in line.lower()
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


def identify_hotspots(annotated_file, threshold_percentage):
    """
    Identifies hotspots in the annotated Callgrind output file based on a given percentage threshold.
    
    Parameters:
    - annotated_file (str): The path to the annotated Callgrind output file.
    - threshold_percentage (float): The minimum percentage of total calls to consider a function a hotspot.
    
    Returns:
    - list of tuples: Each tuple contains (function_name, call_count).
    """
    call_counts = []
    total_calls = 0

    # First pass: collect all call counts and sum total calls
    with open(annotated_file, 'r', encoding='utf-8') as file:
        for line in file:
            if 'calls=' in line:
                parts = line.split()
                for part in parts:
                    if part.startswith('calls='):
                        call_count = int(part.split('=')[1].replace(',', ''))
                        total_calls += call_count
                        # Extract function name from the line
                        func_name = line.split(' * ')[-1].strip()
                        call_counts.append((func_name, call_count))
                        break

    # Calculate threshold count based on percentage
    threshold_count = total_calls * (threshold_percentage / 100.0)

    # Second pass: filter functions above threshold count
    hotspots = [(func, count) for func, count in call_counts if count >= threshold_count]

    return hotspots


def get_hotspot_functions(annotated_file, threshold):
    """
    Gets a list of hotspot functions from the annotated Callgrind output file.
    
    Parameters:
    - annotated_file (str): The path to the annotated Callgrind output file.
    - threshold (int): The minimum number of calls to consider a function a hotspot.
    
    Returns:
    - list of str: hotspot function files & function source code
    """
    hotspots = identify_hotspots(annotated_file, threshold)
    return [func for func, count in hotspots]

def test():
    # Example usage
    project_dir = 'datasets/quantpp'
    executable_path = f'{project_dir}/quant'
    executable_args = [f'{project_dir}/data/table.csv']
    callgrind_output = f'{project_dir}/callgrind.out.test'
    annotated_output = f'{project_dir}/annotated_callgrind.out'
    cleaned_output = f'{project_dir}/cleaned_callgrind.txt'
    threshold = 50  # Example threshold

    run_under_callgrind(executable_path, callgrind_output, executable_args)
    annotate_callgrind_output(callgrind_output, annotated_output)
    # clean_callgrind_output(annotated_output)
    # hotspots = get_hotspot_functions(cleaned_output, threshold)
    # print("Identified Hotspots:")
    # for hotspot in hotspots:
    #     print(hotspot)


if __name__ == "__main__":
    test()