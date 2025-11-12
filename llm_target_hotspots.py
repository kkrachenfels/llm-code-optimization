import argparse
from openai import OpenAI

from config import API_KEY
from prepare_prompts import create_prompts

class HotspotOptimizer:
    def __init__(self, project_dir):
        self.client = OpenAI(api_key = API_KEY)
        self.project_dir = project_dir
        self.prompts = create_prompts(self.project_dir)
        self.responses = []

    def query_llm(self, all_prompts=False, prompt_no=2) -> str:
        if all_prompts:
            for i, prompt in enumerate(self.prompts):
                self.responses.append(self.client.chat.completions.create(
                    model="gpt-5-nano",  # nano for now to save $$$ lol
                    messages=[{"role": "user", "content": prompt}],
                ))
                self.responses[i] = self.responses[i].choices[0].message.content
        
            print(self.responses[i])
        else:
            self.responses.append(self.client.chat.completions.create(
                model="gpt-5-nano",  # nano for now to save $$$ lol
                messages=[{"role": "user", "content": self.prompts[prompt_no]}],
                ).choices[0].message.content
            )
            print(self.responses[0])
        
        return self.responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="match callgrind hotspots to locations in source code, craft prompt")
    parser.add_argument('-d', '--project-dir', type=str, default="datasets/quantpp", help="Path to the project directory where profiling info is stored")
    parser.add_argument('-n', '--prompt_no', type=int, default=0)
    args = parser.parse_args()

    h = HotspotOptimizer(args.project_dir)
    h.query_llm(prompt_no=args.prompt_no)