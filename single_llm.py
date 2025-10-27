import openai

from config import API_KEY

class HotspotOptimizer:
    def __init__(self):
        openai.api_key = API_KEY

    def query_llm(self, prompt: str) -> str:
        response = openai.Completion.create(
            engine="gpt-5-nano", ## nano for now to save $$$ lol
            prompt=prompt,
            # max_tokens=1024,
            # temperature=0.7,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
        )
        return response.choices[0].text.strip()
    

def optimize_hotspot(self, hotspot: str, node: str, calls_hotspot: int, calls_node: int,
                        hotspot_calls_node: int, node_calls_hotspot: int,
                        hotspot_code_snippets: list[str], node_code_snippets: list[str],
                        class_code_snippets: list[str], strategy: str) -> str: