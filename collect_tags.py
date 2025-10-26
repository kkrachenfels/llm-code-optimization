import os
import argparse
import subprocess

from time import sleep


def make_ctags_file(source_dir):
    """
    Generates a ctags file for the given project source directory.
    
    Parameters:
    - source_dir (str): The directory containing project source files.
    """
    # Ensure the output directory exists
    if not os.path.isdir(source_dir):
        print(f"[!] {source_dir} is not a valid directory or doesn't exist.")
        return
       
    # Run the ctags command
    # ctags CLI options:
    # https://docs.ctags.io/en/latest/man/ctags.1.html#id1
    subprocess.run([
        'ctags',
        '-R',
        '--exclude=Makefile',
        '-h',
        '.h.H.hh.hpp.hxx.h.c.cpp.cxx',
        '--format=2',
        '--sort=1',
        '--fields=nKs',
        '-f',
        f'{source_dir}/tags',
        source_dir])

def read_ctags_file(file_path):
    """
    Reads a ctags file and returns a list of tags and information.
    
    Parameters:
    - file_path (str): The path to the ctags file.
    """
    # Define column names for the ctags file
    column_names = ['tag_name', 'file_name', 'tag_declaration', 'tag_type', 'line_n', 'extra_info']
    tags = [column_names]

    with open(file_path, 'r') as f:
        while (line := f.readline()):
            tag = []
            # Skip header lines starting with '!'
            if line.startswith('!'):
                continue
            line = line.replace("/^", "").replace("$/;\"", "").strip("\n")
            print(line)
            tag += line.split('\t')
            for i in range(len(tag)):
                try:
                    if tag[i].strip() == '':
                        tag[i:] = tag[i+1:]
                except:
                    break
            for i in range(len(tag)):
                tag[i] = tag[i].strip().strip(';{:')
                if i == 4:
                    try:
                        tag[i] = int((tag[i]).replace("line:", ""))
                    except:
                        pass
            tag += [''] * (6 - len(tag))  # Pad to ensure complete rows
            tags.append(tag)
    
    output_file = file_path.replace('tags', 'tags.tsv')
    with open(output_file, 'w') as f:
        for row in tags:
            for i, item in enumerate(row):
                if i != 0:
                    f.write(f"\t{str(item)}")
                else:
                    f.write(f"{str(item)}")
            f.write('\n')

    return tags

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and read ctags files for a project.")
    parser.add_argument('-d', '--source_dir', type=str, default="datasets/quantpp", help="Directory containing project source files.")
    args = parser.parse_args()
    
    make_ctags_file(args.source_dir)
    sleep(0.5)  # Ensure ctags file is written before reading
    read_ctags_file(f"{args.source_dir}/tags")