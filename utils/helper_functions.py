import yaml, sys
from pathlib import Path

root_path = str(Path(__file__).resolve().parents[1])
if root_path not in sys.path:
    sys.path.append(root_path)

def retrieve_prompt(agent_name: str) -> str:
    with open("prompts/fundamental_analysis/fundamental_analysis_agent.yaml", "r") as file: 
        prompts = yaml.safe_load(file)
        return prompts[agent_name]["prompt"]