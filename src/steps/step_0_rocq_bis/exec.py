import os
import argparse
import json
import re
import shutil
from collections import defaultdict
import concurrent.futures
import logging
logger = logging.getLogger(__name__)

from tqdm import tqdm
from src.training.eval import eval_tactics, start_pet_server, stop_pet_server, timeout, TimeoutError
"""
First step bis: Check extract steps and source code are compilable using Pytanque. Retrieve (sub)goals from pytanque.
"""

def check_list(data_list, port=8765):
    server_process = start_pet_server(mean_wait=1, port=port)
    first_eval_tactics = timeout(seconds=60)(eval_tactics)
    second_eval_tactics = timeout(seconds=10)(eval_tactics)
    first_tactic = True
    for k, (data, data_path) in list(enumerate(data_list)):
        name_thm = data['name']
        workspace = os.path.abspath(data['workspace'])
        filepath = data['filepath']
        tactics = data['steps']
        try:
            if first_tactic:
                goal_init, res = first_eval_tactics(name_thm, workspace, filepath, tactics, port=port)
                first_tactic = False
            else:
                goal_init, res = second_eval_tactics(name_thm, workspace, filepath, tactics, port=port)
        except TimeoutError as e:
            print(e)
            stop_pet_server(server_process)
            server_process = start_pet_server(mean_wait=1, port=port)
            first_tactic = True
            os.remove(data_path)
            continue

        if res[-1]['status'] != 'finish':
            logger.warning(f'{data_path} does not compile using Pytanque. Ignore the file.')
            logger.warning(f'Last result: {res[-1]}')
            os.remove(data_path)
            continue
        
        if 'coqpyt_aux' in data['term']:
            logger.warning(f'{data_path} term seems to contain trace of partial compilation.')
            os.remove(data_path)
            continue
        data['evaluation'] = res
        data['goals'] = [goal_init] + [entry['goals'] for entry in res]
        data['pytanque_check'] = True
        with open(data_path, 'w') as file_io:
            json.dump(data, file_io, indent=4)
        
        if k%1000 == 999:
            stop_pet_server(server_process)
            server_process = start_pet_server(mean_wait=1, port=port)
            first_tactic = True
    stop_pet_server(server_process)

def copy_if_not_exists(src, dst):
    if os.path.exists(dst):
        return
    shutil.copy2(src, dst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/steps/step_0/', help='Output path from previous step')
    parser.add_argument('--output', default='export/steps/step_0_bis/', help='Output path')
    parser.add_argument('--max-workers', default=8, type=int, help='Number of workers')

    args = parser.parse_args()

    # shutil.copytree(args.input, args.output, copy_function=copy_if_not_exists, dirs_exist_ok=True)

    pattern = r'(term_[0-9]+).json'
    to_do = defaultdict(list)
    for root, _, files in os.walk(args.output):
        for filename in files:
            match = re.match(pattern, filename)
            if match:
                data_path = os.path.join(root, filename)
                with open(data_path, 'r') as file_io:
                    data = json.load(file_io)
                if 'pytanque_check' in data:
                    continue
                to_do[root].append((data, data_path))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for k, source in enumerate(to_do, start=1):
            futures.append(executor.submit(check_list, to_do[source], port=8765 + k))
        
        for _ in tqdm(concurrent.futures.as_completed(futures), desc="Overall progress", position=0, total=len(futures)):
            pass