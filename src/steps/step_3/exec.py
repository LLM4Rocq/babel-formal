import os
import random
import argparse
import time
import random
import json
import re
import concurrent.futures
import shutil
from copy import deepcopy

import yaml
from openai import OpenAI
from tqdm import tqdm

from tqdm import tqdm

"""
Fourth step: Generate reasoning using deepseek R1.
"""

def generate_output(prompt, client, config):
    """
    Sends prompt to client using config.
    """
    completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        **config
    )
    return {"reasoning": completion.choices[0].message.reasoning, "content": completion.choices[0].message.content}

def process_prompt(prompt, export_path, data, client, config, num_gen=10, delay=0):
    """
    Executes multiple generation of the same prompt, export them sequentially.
    """
    time.sleep(delay)
    for k in range(num_gen):
        output_entry = generate_output(prompt, client, config)
        data["reasonings"].append(output_entry)
        with open(export_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Saved_{k}: {export_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/step_2/', help='Input dataset path')
    parser.add_argument('--output', default='export/step_3/', help='Output dataset path')
    parser.add_argument('--num_gen', default=1, type=int, help='Number of reasoning to generate per term')
    parser.add_argument('--max_workers', default=100, type=int, help='Max number of concurrent workers')
    parser.add_argument('--mean_delay', default=10, type=int, help='Mean delay before a request is send: use this parameter to load balance')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.txt')
    with open(prompt_path, 'r') as file:
        prompt_template = file.read()

    client = OpenAI(
        base_url=config['base_url'],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    shutil.copytree(args.input, args.output, dirs_exist_ok=True)
    
    to_do = []
    pattern = r'(term_[0-9]+).json'
    for root, _, files in os.walk(args.output):
        for file in files:
            match = re.match(pattern, file)
            if match:
                term_name = match.group(1)
                filepath = os.path.join(root,file)

                with open(filepath, 'r') as file:
                    data = json.load(file)
                
                data_template = deepcopy(data)
                for entry in data_template:
                    if isinstance(data[entry], list):
                        data_template[entry] = "\n".join(data_template[entry])

                if 'reasonings' not in data:
                    data['reasonings'] = []
                assert isinstance(data['reasonings'], list), "Reasonings entry should be a list, issue with inputs"
                prompt = prompt_template.format(**data_template)
                to_do.append((prompt, filepath, data))
    delay_max = args.mean_delay*2
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:  # Adjust the number of workers as needed
        futures = []
        futures += [executor.submit(process_prompt, prompt, export, entry, client, config['request_config'], num_gen=args.num_gen, delay=random.randint(0, delay_max)) for prompt, export, entry in to_do]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass




