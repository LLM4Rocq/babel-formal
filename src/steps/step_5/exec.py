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

def delete_empty_folders(root):
    deleted = set()
    for current_dir, subdirs, files in os.walk(root, topdown=False):
        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break
    
        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)

def generate_output(prompt, client, config):
    completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        **config
    )
    return {"reasoning": completion.choices[0].message.reasoning, "content": completion.choices[0].message.content}

def process_prompt(prompts, export_path, data, client, config, delay=0):
    time.sleep(delay)
    for prompt, reasoning, score in zip(prompts, data['reasonings'], data['scores']):
        output = generate_output(client, prompt, config)
        if 'boxed{yes}' in output['content']:
            print(f"Good reasoning: {export_path}")
            data['reasoning'] = reasoning
            data['score'] = score
            with open(export_path, 'w') as file:
                json.dump(data, file)
            return ()
        else:
            print(f"Bad reasoning: {export_path}")
    
    os.remove(export_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/step_4/', help='Input dataset path')
    parser.add_argument('--output', default='export/step_5/', help='Output dataset path')
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

                if 'reasoning' not in data:
                    prompts = []

                    score_reasonings = [(s['score_decision'], r) for s,r in zip(data['scores'], data['reasonings'])]
                    score_reasonings = sorted(score_reasonings, reverse=True)
                    for _, reasoning in score_reasonings:
                        prompt = prompt_template.format(reasoning=reasoning, **data_template)
                        prompts.append(prompt)
                    to_do.append((prompts, filepath, data))
    delay_max = args.mean_delay*2
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:  # Adjust the number of workers as needed
        futures = []
        futures += [executor.submit(process_prompt, prompt, export, entry, client, config['request_config'], delay=random.randint(0, delay_max)) for prompt, export, entry in to_do]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
    
    delete_empty_folders(args.output)




