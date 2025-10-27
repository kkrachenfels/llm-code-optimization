import os
import csv
import json
import argparse
from thefuzz import fuzz # use fuzzy matching for fns between callgrind & ctags

def read_hotspots(hotspots_json):
    hotspots = []
    with open(hotspots_json, 'r') as f:
        hotspots = json.load(f)
    print(hotspots)
    return hotspots

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


def match_hotspot_to_tag(h, tags):
    hotspot_matches = []
    full_fn_name = h['name']
    full_fn_name = '['.join(full_fn_name.split('[')[:-1])
    fn_name = '('.join(full_fn_name.split('(')[:-1]).strip()

    for t_name, t_info in tags.items():
        if fn_name in t_name and t_info['tag_type'] == 'function':
            fuzzy_match_score = fuzz.partial_ratio(full_fn_name, t_info['tag_declaration'])
            print(f"{fn_name}")
            print(fuzzy_match_score)
            print(f"\t[found matching tag]: {t_name}")
            print(f"\t[tag info]: file_name: {t_info['file_name']}, line_n: {t_info['line_n']}")
            print(f"\t[tag_declaration]: {t_info['tag_declaration']}")
            hotspot_matches.append({
                'fn_name': fn_name,
                'full_fn_name': full_fn_name,
                'file': t_info['file_name'],
                'line_n': t_info['line_n'],
                'fn_decl': t_info['tag_declaration'],
                'fuzzy_match_score': fuzzy_match_score
            })
    # print(hotspot_matches)  
    return hotspot_matches
    

def find_hotspot_sources(hotspot_match):
    lines = []
    try:
        with open(hotspot_match['file'], 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[!] Unable to open source file {hotspot_match['file']}")
    
    start_line_n = int(hotspot_match['line_n'])-1
    end_line_n = None
    # print(lines[start_line_n].strip())
    open_curly_braces = 0
    for i, line in enumerate(lines[start_line_n:]):
        if '{' in line:
            open_curly_braces += 1
        if '}' in line:
            open_curly_braces -= 1
        if i > 0 and not open_curly_braces:
            # print(i)
            end_line_n = start_line_n + i
            break

    fn_lines = "".join(lines[start_line_n: end_line_n + 1])
    print(fn_lines)
    return fn_lines


def build_prompt(hotspot: str, hotspot_code_snippet: str, 
                    caller_nodes: list[str], caller_nodes_times: list[int], caller_code_snippets: list[str],
                    callee_nodes: list[str], callee_nodes_times: list[int], callee_code_snippets: list[str],
                    # node_code_snippets: list[str], class_code_snippets: list[str], 
                    # strategy: str
                    ) -> str:
    prompt = f"""
Task: As a programmer, you need to optimize the hotspot {hotspot}.
Use a Chain-of-Thought approach to understand the code and its contexts,
and then optimize the given hotspot function 

Context:
When the project is running:
    """

    for i, (node, times) in enumerate(zip(caller_nodes, caller_nodes_times)):
        prompt += f"- The function {node} calls hotspot {times} times:\n"
        if caller_code_snippets[i]:
            prompt += "```code snippet```\n"
            prompt += caller_code_snippets[i]
            prompt += "```code snippet```\n"

    for i, (node, times) in enumerate(zip(callee_nodes, callee_nodes_times)):
        prompt += f"- The hotspot calls {node} function {times} times:\n"
        if callee_code_snippets[i]:
            prompt += "```code snippet```\n"
            prompt += callee_code_snippets[i]
            prompt += "```code snippet```\n"
    
    prompt += f"""
- Based on these contexts, optimize the hotspot function {hotspot}:
```code snippet```
{hotspot_code_snippet if hotspot_code_snippet else ''}
```code snippet```
Here is the response template:
## Optimized hotspot function:
## Affected functions:
## Optimization strategy:
    """

    """
    # strategy information

    During the code static analysis phase:
    - The hotspot references {node}: The {node} references hotspot: Class {node} that hotspot belongs to:
    ```code snippet```
    {node_code_snippets[0] if node_code_snippets else ''}
    ```code snippet```
    {node_code_snippets[1] if len(node_code_snippets) > 1 else ''}
    ```code snippet```
    {class_code_snippets[0] if class_code_snippets else ''}
    
    """

    print(prompt)
    with open('prompt.md', 'w') as f:
        f.write(prompt)


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
    
    hotspots = read_hotspots(json_path)
    print(f"{'=' * 50}")
    ctags = read_ctags(ctags_path)
    print(f"{'=' * 50}")

    for hotspot in hotspots:
        prompt_dict = {}

        # take first match for now,
        # can take highest fuzzy match later
        hs_match = match_hotspot_to_tag(hotspot['info'], ctags)[0]
        print(f"{'=' * 50}")
        hs_source = find_hotspot_sources(hs_match)

        prompt_dict['hotspot'] = hs_match['fn_name']
        prompt_dict['hotspot_code_snippet'] = hs_source

        for callstack_fn in hotspot["callstack"]:

            if callstack_fn['type'] == '*':
                # hotspot fn itself, already matched
                continue

            # matching fns that call the hotspot:
            print(f"Matching callstack fn {callstack_fn['name'][:60]}...")
            try:
                cs_match = match_hotspot_to_tag(callstack_fn, ctags)[0]
                cs_source = find_hotspot_sources(cs_match)
            except IndexError:
                cs_match = {'fn_name': callstack_fn['name']}
                cs_source = None

            if callstack_fn['type'] == '<':
                # caller:
                if 'caller_nodes' in prompt_dict:
                    prompt_dict['caller_nodes'].append(cs_match['fn_name'])
                    prompt_dict['caller_nodes_times'].append(callstack_fn['num_calls'])
                    prompt_dict['caller_code_snippets'].append(cs_source)
                else:
                    prompt_dict['caller_nodes'] = [cs_match['fn_name']]
                    prompt_dict['caller_nodes_times'] = [callstack_fn['num_calls']]
                    prompt_dict['caller_code_snippets'] = [cs_source]

            else:
                # callee
                if 'callee_nodes' in prompt_dict:
                    prompt_dict['callee_nodes'].append(cs_match['fn_name'])
                    prompt_dict['callee_nodes_times'].append(callstack_fn['num_calls'])
                    prompt_dict['callee_code_snippets'].append(cs_source)
                else:
                    prompt_dict['callee_nodes'] = [cs_match['fn_name']]
                    prompt_dict['callee_nodes_times'] = [callstack_fn['num_calls']]
                    prompt_dict['callee_code_snippets'] = [cs_source]

            print(f"{'=' * 50}")

    build_prompt(
        prompt_dict['hotspot'],
        prompt_dict['hotspot_code_snippet'],
        prompt_dict['caller_nodes'],
        prompt_dict['caller_nodes_times'],
        prompt_dict['caller_code_snippets'],
        prompt_dict['callee_nodes'],
        prompt_dict['callee_nodes_times'],
        prompt_dict['callee_code_snippets'],
    )
        
