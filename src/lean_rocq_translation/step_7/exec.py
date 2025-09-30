import os
import random
import argparse
import time
import random
import json
import concurrent.futures

import yaml
from openai import OpenAI
from tqdm import tqdm

from tqdm import tqdm

"""
Seventh step: Generate reasonings traces for Lean 4 using Gemini 2.5 pro.
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

def process_prompt(prompt, export_path, data, client, config, delay=0):
    """
    Executes multiple generation of the same prompt, export them sequentially.
    """
    time.sleep(delay)
    output = generate_output(prompt, client, config)
    data["reasoning"] = output
    with open(export_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-step-3', default='export/dataset/step_3/step_3.json', help='Input dataset path')
    parser.add_argument('--input-step-6', default='export/dataset/step_6/step_6.json', help='Input dataset path')
    parser.add_argument('--output', default='export/dataset/step_7/', help='Output dataset path')
    parser.add_argument('--max-workers', default=25, type=int, help='Max number of concurrent workers')
    parser.add_argument('--mean-delay', default=10, type=int, help='Mean delay before a request is send: use this parameter to load balance')
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

    to_do = []
    with open(args.input_step_6, 'r') as file:
        content = json.load(file)
    
    with open(args.input_step_3, 'r') as file:
        step_p = json.load(file)
    name_to_step = {}
    for entry in step_p:
        name_to_step[entry['name']] = entry

    for entry in content:
        dependencies = []
        for symbol in entry['symbols']:
            if symbol['category'] in ['def', 'class', 'constructor', 'inductive']:
                dependencies.append(symbol['print_text'])
            else:
                dependencies.append(symbol['check_text'])
        
        dependencies = '\n'.join(dependencies)
        statement = entry["header"]
        proof = entry["proof"]
        entry["lean_extract"] = name_to_step[entry['name']]['lean_extract']
        term = entry['lean_extract']['term']

        prompt = prompt_template.format(term=term, proof=proof, dependencies=dependencies)
        export_path = os.path.join(args.output, entry['name']+'.json')

        if not os.path.exists(export_path):
            to_do.append((prompt, export_path, entry))

        # if os.path.exists(export_path):
        #     with open(export_path, 'r') as file:
        #         content = json.load(file)
            
    delay_max = args.mean_delay*2
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:  # Adjust the number of workers as needed
        futures = []
        futures += [executor.submit(process_prompt, prompt, export, entry, client, config['request_config'], delay=random.randint(0, delay_max)) for prompt, export, entry in to_do]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass




