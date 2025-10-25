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
                tag[i] = tag[i].strip().strip(';')
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


read_ctags_file('datasets/quantpp/tags')