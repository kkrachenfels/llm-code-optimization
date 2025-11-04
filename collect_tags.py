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
        '--exclude=*Makefile,',
        '--exclude=*.html',
        '--exclude=*.js',
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
            tag = [''] * 6
            # Skip header lines starting with '!'
            if line.startswith('!'):
                continue
            
            line_split = line.split("/^")
            tag_other_info = line_split[1].split("/;\"")
            tag_info = tag_other_info[0]
            tag[2] = tag_info.strip(';{:$').strip().replace('\t', ' ')

            tag_first_parts = line_split[0].split('\t')
            tag[0] = tag_first_parts[0]
            tag[1] = tag_first_parts[1]

            tag_last_parts = tag_other_info[1].strip().strip('\t').split('\t')
            tag[3] = tag_last_parts[0]
            tag[4] = tag_last_parts[1]
            if len(tag_last_parts) > 2:
                tag[5] = tag_last_parts[2]

            try:
                tag[4] = int((tag[4]).replace("line:", ""))
            except:
                print(f"[!] Unable to get line # from tag {tag[0]} with line {tag[4]}")
            
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