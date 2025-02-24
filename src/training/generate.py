import argparse
import time
import json
import os
import yaml
import json
import random
from copy import deepcopy
import concurrent.futures

from tqdm import tqdm
from openai import OpenAI
from pytanque import Pytanque, PetanqueError

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

def process_prompt(prompt, entry, export_path, client, config, delay=0, num_gen=4):
    """
    Executes multiple generation of the same prompt, export them sequentially.
    """
    time.sleep(delay)
    for k in range(num_gen):
        output_entry = generate_output(prompt, client, config)
        output_entry['category'] = entry['category']
        output_entry['name'] = entry['name']
        with open(export_path + f'_sample_{k}.json', 'w') as file:
            json.dump(output_entry, file, indent=4)
        print(f"Saved {k}: {export_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dataset', default='export/steps/final.json', help='Dataset')
    parser.add_argument('--input-sources', default='export/steps/sources', help='Directory containing sources files')
    parser.add_argument('--output', default='export/eval', help='Output directory')
    parser.add_argument('--dataset-entry', default='test', help='Entry to use in the dataset')
    parser.add_argument('--max-workers', default=100, type=int, help='Max number of concurrent workers')
    parser.add_argument('--mean-delay', default=10, type=int, help='Mean delay before a request is send: use this parameter to load balance')
    parser.add_argument('--k', default=4, help='Parameter k of pass@k')
    parser.add_argument('--config', default='src/training/config/o3minihigh.yaml', help='Config file to evaluate model')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    prompt_path = os.path.join(os.path.dirname(__file__), config['prompt'])
    with open(prompt_path, 'r') as file:
        prompt_template = file.read()

    with open(args.input_dataset, 'r') as file:
        dataset = json.load(file)

    client = OpenAI(
        base_url=config['base_url'],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    to_do = []
    
    for entry in dataset[args.dataset_entry]:

        prompt = prompt_template.format(term=entry['term'], constants='\n'.join(entry['constants']), notations='\n'.join(entry['notations']))
        to_do.append((prompt, entry, os.path.join(args.output, 'term_' + entry['name'].lower())))

    random.shuffle(to_do)
    to_do = to_do[:50]
    delay_max = args.mean_delay*2
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:  # Adjust the number of workers as needed
        futures = []
        futures += [executor.submit(process_prompt, prompt, entry, export, client, config['request_config'], delay=random.randint(0, delay_max), num_gen=args.k) for prompt, entry, export in to_do]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
