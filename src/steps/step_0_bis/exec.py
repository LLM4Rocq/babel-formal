import os
import argparse
import json
import re
import shutil
import logging
logger = logging.getLogger(__name__)

from tqdm import tqdm
from src.training.eval import eval_tactics

"""
First step bis: Check extract steps and source code are compilable using Pytanque. Retrieve (sub)goals from pytanque.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/step_0/', help='Output path from previous step')
    parser.add_argument('--output', default='export/step_0_bis/', help='Output path')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        shutil.copytree(args.input, args.output, dirs_exist_ok=False)

    pattern = r'(term_[0-9]+).json'
    to_do = []
    for root, subdirs, files in os.walk(args.output):
        for filename in files:
            match = re.match(pattern, filename)
            if match:
                data_path = os.path.join(root, filename)
                source_path = os.path.join(root, 'source.v')
                to_do.append((data_path, source_path))

    for data_path, source_path in tqdm(to_do):
        with open(data_path, 'r') as file_io:
            data = json.load(file_io)
        
        if 'pytanque_check' in data:
            continue
        name_thm = data['name']
        tactics = [tactic for tactic,_,_ in data['steps'] if 'Proof.' not in tactic and 'Qed.' not in tactic]
        
        goal_init, res = eval_tactics(name_thm, source_path, tactics)

        if res[-1]['status'] != 'finish':
            logger.warning(f'{data_path} does not compile using Pytanque. Ignore the file.')
            logger.warning(f'Last result: {res[-1]}')
            os.remove(data_path)
            continue
        
        if 'coqpyt_aux' in data['term']:
            logger.warning(f'{data_path} term seems to contain trace of partial compilation.')
            os.remove(data_path)
            continue

        data['goals'] = [goal_init] + [entry['goals'] for entry in res]
        data['pytanque_check'] = True
        with open(data_path, 'w') as file_io:
            json.dump(data, file_io, indent=4)
        