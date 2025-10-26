import os
import csv
import json
import argparse


def read_hotspots(hotspots_json):
    hotspots = []
    with open(hotspots_json, 'r') as f:
        hotspots = json.load(f)
    print(hotspots)

def read_ctags(ctags_tsv):
    tsv = {}
    headers = []
    with open(ctags_tsv, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        for i, row in enumerate(csv_reader):
            row_data = {}
            if i == 0:
                headers = row
            else:
                for h, r in zip(headers, row):
                    row_data[h] = r
                # only get function tags for now
                if row_data['tag_type'] == 'function':
                    tsv[row_data['tag_name']] = row_data
    print(tsv)
    return tsv


def match_hotspot_to_tag():
    pass


def find_hotspot_source():
    pass


def build_prompt(hotspot: str, node: str, calls_hotspot: int, calls_node: int,
                    hotspot_calls_node: int, node_calls_hotspot: int,
                    hotspot_code_snippets: list[str], node_code_snippets: list[str],
                    class_code_snippets: list[str], strategy: str) -> str:
    prompt = f"""
    Task: As a programmer, you need to optimize the hotspot {hotspot}.
    Use a Chain-of-Thought approach to understand the code and its contexts,
    and then optimize the given hotspot function {strategy}
    
    Context:
    When the project is running:
    - The hotspot calls {node} {hotspot_calls_node} times: {calls_hotspot}
    - The {node} calls hotspot {node_calls_hotspot} times:
    ```code snippet```
    {hotspot_code_snippets[0] if hotspot_code_snippets else ''}
    ```code snippet```
    {hotspot_code_snippets[1] if len(hotspot_code_snippets) > 1 else ''}
    
    During the code static analysis phase:
    - The hotspot references {node}: The {node} references hotspot: Class {node} that hotspot belongs to:
    ```code snippet```
    {node_code_snippets[0] if node_code_snippets else ''}
    ```code snippet```
    {node_code_snippets[1] if len(node_code_snippets) > 1 else ''}
    ```code snippet```
    {class_code_snippets[0] if class_code_snippets else ''}
    - Based on these contexts, optimize the hotspot function {hotspot}:
    ```code snippet```
    {hotspot_code_snippets[0] if hotspot_code_snippets else ''}
    Here is the response template:
    ## Optimized hotspot function:
    ## Affected functions:
    ## Optimization strategy:
    """

    print(prompt)
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="match callgrind hotspots to locations in source code, craft prompt")
    parser.add_argument('-d', '--project-dir', type=str, default="datasets/quantpp", help="Path to the project directory where profiling info is stored")
    args = parser.parse_args()

    if not os.path.isdir(args.project_dir):
        print(f"[!] {args.project_dir} is not a valid directory or doesn't exist.")
        exit()
    
    json_path = f"{args.project_dir}/hotspots.json"

    if not os.path.exists(json_path):
        print(f"[!] {json_path} doesn't exist. Run callgrind.py first.")
        exit()

    ctags_path = f"{args.project_dir}/tags.tsv"

    if not os.path.exists(ctags_path):
        print(f"[!] {ctags_path} doesn't exist. Run collect_tags.py first.")
        exit()
    
    read_hotspots(json_path)
    print(f"{'=' * 20}")
    read_ctags(ctags_path)

