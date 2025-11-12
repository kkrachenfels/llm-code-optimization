import json
import argparse
from openai import OpenAI

from config import API_KEY
from prepare_prompts import create_prompts

class HotspotOptimizer:
    def __init__(self, project_dir, provide_hotspots=False):
        self.client = OpenAI(api_key = API_KEY)
        self.project_dir = project_dir
        self.prompts = []
        self.provide_hotspots = provide_hotspots
        self.responses = []
        if not self.provide_hotspots:
            self.prompts.append(self.read_prompt())
        else:
            self.create_prompt()

    def read_prompt(self):
        with open("prompts/files_no_hotspots.md", 'r') as f:
            return f.read()

    def create_prompt(self):
        print("creating prompt")
        base_prompt = ""
        with open("prompts/files_no_hotspots.md", 'r') as f:
            base_prompt = f.read()
        
        print(len(base_prompt))

        with open(self.project_dir + "/hotspots.json", 'r') as f:
            hotspots = f.read()

        hotspot_prompt = f"""
First, information about top hotspot functions obtained from callgrind is listed below:

```json
{hotspots}
```

Now here's the source tree of the project:
        """

        full_prompt = base_prompt.split("Source Tree:")[0]
        full_prompt += hotspot_prompt
        full_prompt += "Source Tree:\n"
        full_prompt += base_prompt.split("Source Tree:")[1]
        
        print(full_prompt)
        self.prompts.append(full_prompt)

    def query_llm(self) -> str:
        self.responses.append(self.client.chat.completions.create(
            model="gpt-5-mini",  # mini for now to save $$$ lol
            messages=[{"role": "user", "content": self.prompts[0]}],
            ).choices[0].message.content
        )
        print(self.responses[0])

        return self.responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="match callgrind hotspots to locations in source code, craft prompt")
    parser.add_argument('-d', '--project-dir', type=str, default="datasets/quantpp", help="Path to the project directory where profiling info is stored")
    parser.add_argument('-p', '--provide-hotspots', action='store_true', help="Provide hotspots to the LLM")
    args = parser.parse_args()

    h = HotspotOptimizer(args.project_dir, args.provide_hotspots)
    responses = h.query_llm()
    for i, response in enumerate(responses):
        if args.provide_hotspots:
            with open(f'responses/response_full_file_hotspots{i}.md', 'w') as f:
                f.write(response)
        else:
            with open(f'responses/response_full_file_{i}.md', 'w') as f:
                f.write(response)