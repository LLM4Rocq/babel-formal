import json
import os
import time
import argparse
import concurrent.futures
import random
import copy

import yaml
from tqdm import tqdm

from src.llm.openai_chat import OpenAIChatLLM
from src.evaluator.factory import make_prover
from src.evaluator.prover import DatasetItem, Prover, MessageType

def extract_proof(block: str):
    if '<BEGIN_ROCQ_PROOF>' in block:
        proof = block.split('<BEGIN_ROCQ_PROOF>')[1]
        return proof.split('</END_ROCQ_PROOF>')[0].strip()
    if '<BEGIN_LEAN_PROOF>' in block:
        proof = block.split('<BEGIN_LEAN_PROOF>')[1]
        return proof.split('</END_LEAN_PROOF>')[0]
    raise Exception('No proof found')

def process_prompt(messages, export_path, data, item: DatasetItem, prover: Prover, client: OpenAIChatLLM, delay=0, bootstrap=4):
    """
    Executes multiple generation of the same prompt, export them sequentially.
    """
    time.sleep(delay)

    data['output'] = []
    data['result'] = []
    data['prompt'] = []
    for _ in range(bootstrap + 1):
        data['prompt'].append(copy.deepcopy(messages))
        try:
            output_entry = client.generate(messages)
            data["output"].append(output_entry)
            proof = extract_proof(output_entry)
            prover.start_thm(item)
            message = prover.check_proof(proof)
            if message.status != MessageType.FINISH:
                error_message = "\n".join(message.message.split('\n')[:3])
                messages.append({
                    "role": "user",
                    "content": f"You already tried this proof:\n{proof}\n but it fails with this error:\n{error_message}."
                })
                data['result'].append(('ERROR/ONGOING', message.message))
            else:
                data['result'].append(('FINISH', message.message))
                break
        except Exception as e:
            data['output'].append(str(e))
            data['result'].append(('Exception', str(e)))
            with open(export_path, 'w') as file:
                json.dump(data, file, indent=4)
    with open(export_path, 'w') as file:
        json.dump(data, file, indent=4)
    return output_entry

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='benchmark/test.json', help='Input dataset path')
    parser.add_argument('--output', default='export/eval/gpt_5_translate/', help='Output dataset path')
    parser.add_argument('--max-workers', default=16, type=int, help='Max number of concurrent workers')
    parser.add_argument('--mean-delay', default=5, type=int, help='Mean delay before a request is send: use this parameter to load balance')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    prompt_l_r_path = os.path.join(os.path.dirname(__file__), 'prompt_l_r.txt')
    prompt_r_l_path = os.path.join(os.path.dirname(__file__), 'prompt_r_l.txt')

    with open(prompt_l_r_path, 'r') as file:
        prompt_l_r_template = file.read()
    with open(prompt_r_l_path, 'r') as file:
        prompt_r_l_template = file.read()

    client = OpenAIChatLLM(
        **config
    )
    
    lean_prover = make_prover('lean', 'dataset/')
    rocq_prover = make_prover('rocq', 'dataset/')

    to_do_rocq = []
    to_do_lean = []

    lr_folder = os.path.join(args.output, "lean_to_rocq")
    rl_folder = os.path.join(args.output, "rocq_to_lean")
    os.makedirs(lr_folder, exist_ok=True)
    os.makedirs(rl_folder, exist_ok=True)

    with open(args.input) as file:
        dataset= json.load(file)
    
    for content in dataset:
        source = content['source']
        for entry in tqdm(content['items'], position=1, desc="Theorems remaining", disable=True):
            name = entry['name']
            line_start, line_end = entry['lean']['lines']

            item_lean = DatasetItem("lean", source, name, lines=(line_start, line_end))
            item_rocq = DatasetItem("rocq", source, name, lines=(0, 0))

            entry_lean = entry['lean']
            entry_rocq = entry['coq']
            
            lean_statement = entry_lean['statement']
            lean_proof = entry_lean['proof']

            rocq_statement = entry_rocq['statement']
            rocq_proof = entry_rocq['proof']

            lean_dependencies_set = set()
            for subentry in entry_lean['dependencies']['statement']:
                lean_dependencies_set.add(subentry['symbol'])
            for subentry in entry_lean['dependencies']['proof']:
                lean_dependencies_set.add(subentry['symbol'])
            
            lean_dependencies = ""
            for symbol in lean_dependencies_set:
                lean_dependencies += content['lean'][symbol]['content'] + '\n'

            rocq_dependencies_set = set()
            for subentry in entry_rocq['dependencies']['statement']:
                rocq_dependencies_set.add(subentry['symbol'])
            for subentry in entry_rocq['dependencies']['proof']:
                rocq_dependencies_set.add(subentry['symbol'])
            
            rocq_dependencies = ""
            for symbol in rocq_dependencies_set:
                rocq_dependencies += content['coq'][symbol]['content'] + '\n'
            
            prompt_lean_to_rocq = prompt_l_r_template.format(lean_statement=lean_statement, lean_proof=lean_proof, rocq_statement=rocq_statement, dependencies=rocq_dependencies)
            prompt_rocq_to_lean = prompt_r_l_template.format(rocq_statement=rocq_statement, rocq_proof=rocq_proof, lean_statement=lean_statement, dependencies=lean_dependencies)
            filepath_l_r = os.path.join(lr_folder, f"{name}.json")
            filepath_r_l = os.path.join(rl_folder, f"{name}.json")

            if not os.path.exists(filepath_r_l):
                to_do_lean.append((prompt_rocq_to_lean, filepath_r_l, item_lean, entry))
            if not os.path.exists(filepath_l_r):
                to_do_rocq.append((prompt_lean_to_rocq, filepath_l_r, item_rocq, entry))

    delay_max = args.mean_delay*2
    random.shuffle(to_do_lean)
    random.shuffle(to_do_rocq)
    for prompt, export, item_lean, entry in tqdm(to_do_lean):
        process_prompt([{"role": "user", "content": prompt}], export, entry, item_lean, lean_prover, client, delay=0)
    
    for prompt, export, item_rocq, entry in tqdm(to_do_rocq):
        process_prompt([{"role": "user", "content": prompt}], export, entry, item_rocq, rocq_prover, client, delay=0)





        