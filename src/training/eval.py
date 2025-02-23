import argparse
import os
import json
import re
from collections import defaultdict

from tqdm import tqdm
from pytanque import Pytanque, PetanqueError

def extract_steps(content):
    pattern = r'\\boxed\{([\s\S]*)\}'
    match = re.search(pattern, content)
    if not match:
        return None
    content = match.group(1)
    instructions = re.findall(r'[^.]+?\.', content)
    return instructions

def eval_tactics(thm, filepath, tactics, url="127.0.0.1", port=8765, timeout=10):
    """
    Try to solve theorem "thm" in the source file "filepath" using tactics.
    """
    with Pytanque(url, port) as pet:
        try:
            state = pet.start(file=filepath, thm=thm)
        except PetanqueError as e:
            return [], [{"status": "error", "goals": [], "message": e.message, "tactic": ""}]
        init_goals = [goal.pp for goal in pet.goals(state)]  
        res = []
        for tactic in tactics:
            entry = {"status": "", "goals": [], "message": "", "tactic": tactic}
            try:
                state = pet.run_tac(state, tactic, verbose=False, timeout=timeout)
                goals = pet.goals(state)

                entry['goals'] = [goal.pp for goal in goals]                
                if state.proof_finished:
                    entry['status'] = "finish"
                else:
                    entry['status'] = "ongoing"
            except PetanqueError as e:
                entry['status'] = "error"
                entry['message'] = e.message
            res.append(entry)
        return init_goals, res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/eval_r1', help='Directory containing evaluations')
    parser.add_argument('--input-sources', default='export/sources', help='Directory containing sources files')
    parser.add_argument('--export', default='export/eval_r1.json', help='Dataset')
    args = parser.parse_args()
    to_do = []
    pattern = r'term_(\S*)_sample_([0-9]+).json'
    for root, _, files in os.walk(args.input):
        for file in files:
            match = re.match(pattern, file)
            if match:
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as file_io:
                    data = json.load(file_io)
                
                data['filename'] = file
                to_do.append((data['category'], data))
    
    to_do = sorted(to_do, key=lambda x:x[0])
    result = defaultdict(list)
    for _, data in tqdm(to_do):
        thm = data['name']
        res = [{"status": "error", "goals": [], "message": "fail", "tactic": ""}]
        if not data['content']:
            res[0]['message'] = 'No content'
            result[thm].append({"evaluation": res, "input": data})
            continue

        steps = extract_steps(data['content'])
        if not steps:
            res[0]['message'] = 'No steps'
            result[thm].append({"evaluation": res, "input": data})
            continue
        source_path = os.path.join(args.input_sources, data['category'], 'source.v')
        _, res = eval_tactics(thm, source_path, steps)
        result[thm].append({"evaluation": res, "input": data})

    with open(args.export, 'w') as file:
        json.dump(result, file, indent=4)
