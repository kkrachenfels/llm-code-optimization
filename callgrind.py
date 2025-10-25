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
    subprocess.run([
        'callgrind_annotate',
        '--autoyes',
        '--tree=both',
        '--inclusive=yes',
        callgrind_file,
        '>',
        annotated_file
    ])


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
