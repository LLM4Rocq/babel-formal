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
from math import exp

import yaml
from openai import OpenAI
from tqdm import tqdm
import numpy as np

"""
Fifth step: Compute score of reasoning based on prediction performance with Qwen 32b R1.
"""

def compute_logprobs(prompt, client, config):
    """
    Send prompt to client using config and retrieve logprobs.
    """
    completion = client.completions.create(
        prompt=prompt,
        extra_body={
            "prompt_logprobs": 0
        }
        **config
    )
    logprobs = completion.choices[0].prompt_logprobs
    return logprobs

def add_hidden_reasonings(data):
    """
    Obfuscate reasoning by words matching (keywords, terms, constants etc.).
    """
    new_data = deepcopy(data)
    pattern = r'(\S*?)\n'
    keywords = ["rewrite", "move", "have", "case", "elim", "by","apply", "=>", "[", "]"]

    for constant in data['constants']:
        match = re.match(pattern, constant)
        if match:
            name = match.group(1)
            keywords.append(name)
    
    hidden_reasonings = []
    for reasoning in data['reasonings']:
        for keyword in keywords:
            reasoning = reasoning.replace(keyword, '#hide')
        
        pattern_1 = r'\S*;'
        for match in re.finditer(pattern_1, reasoning):
            reasoning = reasoning.replace(match.group(0), '#hide')

        pattern_2 = r'(\S*\?\S+)'
        for match in re.finditer(pattern_2, reasoning):
            reasoning = reasoning.replace(match.group(1), '#hide')

        pattern = r'(\S*#hide\S*)'
        for match in re.finditer(pattern, reasoning):
            reasoning = reasoning.replace(match.group(1), '#hide')
        while '#hide #hide' in reasoning:
            reasoning = reasoning.replace('#hide #hide', '#hide')
        while '#hide#hide' in reasoning:
            reasoning = reasoning.replace('#hide#hide', '#hide')
        hidden_reasonings.append(reasoning)
    
    new_data['hidden_reasonings'] = hidden_reasonings
    return new_data

def compute_scores(logprobs):
    """
    Compute average, decision, and completion scores based on given logprobs.
    """
    logprob_avg, logprobs_decision, logprobs_completion = [], [], []
    match_boxed = False
    tot_answer = ""
    for entry in logprobs:
        if not entry:
            continue
        entry = next(iter(entry.values()))
        token = entry['decoded_token']
        logprob = entry['logprob']

        tot_answer += token
        if match_boxed:
            logprob_avg.append(logprob)
            if token[0] == ' ':
                logprobs_decision.append(logprob)
            else:
                logprobs_completion.append(logprob)
        else:
            if '<｜Assistant｜> \\boxed{' in tot_answer:
                match_boxed = True
    
    logprob_avg = np.array(logprob_avg)
    logprobs_decision = np.array(logprobs_decision)
    logprobs_completion = np.array(logprobs_completion)
    return {"score_avg": exp(logprob_avg.mean())*100, "score_decision": exp(logprobs_decision.mean())*100, "score_completion": exp(logprobs_completion.mean())*100}

def process_prompt(prompts, export_path, data, client, config, delay=0):
    """
    Executes multiple generation of the same prompt, export them sequentially.
    """
    time.sleep(delay)
    for k, prompt in enumerate(prompts):
        logprob = compute_logprobs(prompt, client, config)
        data["scores"].append(compute_scores(logprob))
        with open(export_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Saved {k}: {export_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/steps/step_3/', help='Input dataset path')
    parser.add_argument('--output', default='export/steps/step_4/', help='Output dataset path')
    parser.add_argument('--max-workers', default=1, type=int, help='Max number of concurrent workers')
    parser.add_argument('--mean-delay', default=0, type=int, help='Mean delay before a request is send: use this parameter to load balance')
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
                for entry in ['notations', 'constants']:
                    data_template[entry] = "\n".join(data_template[entry])
                
                data = add_hidden_reasonings(data)

                # in case of resuming, 'scores' and 'logprobs' keys already exist
                if 'scores' not in data:
                    data['scores'] = []
                len_scores = len(data['scores'])
                prompts = []
                for hidden_reasoning in data['hidden_reasonings'][len_scores:]:
                    prompt = prompt_template.format(hidden_reasoning=hidden_reasoning, **data_template)
                    prompts.append(prompt)
                to_do.append((prompts, filepath, data))
    delay_max = args.mean_delay*2
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:  # Adjust the number of workers as needed
        futures = []
        futures += [executor.submit(process_prompt, prompt, export, entry, client, config['request_config'], num_gen=args.num_gen, delay=random.randint(0, delay_max)) for prompt, export, entry in to_do]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass




