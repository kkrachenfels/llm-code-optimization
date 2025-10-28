import argparse
from openai import OpenAI

from config import API_KEY
from prepare_prompt import create_prompt

class HotspotOptimizer:
    def __init__(self, project_dir):
        self.client = OpenAI(api_key = API_KEY)
        self.project_dir = project_dir
        self.prompt = create_prompt(self.project_dir)

    def query_llm(self) -> str:
        self.response = self.client.chat.completions.create(
            model="gpt-5-nano",  # nano for now to save $$$ lol
            messages=[{"role": "user", "content": self.prompt}],
        )
        
        print(self.response.choices[0].message.content)
        return self.response.choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="match callgrind hotspots to locations in source code, craft prompt")
    parser.add_argument('-d', '--project-dir', type=str, default="datasets/quantpp", help="Path to the project directory where profiling info is stored")
    args = parser.parse_args()

    h = HotspotOptimizer(args.project_dir)
    h.query_llm()