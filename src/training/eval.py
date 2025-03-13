import argparse
import os
import json
import re
from collections import defaultdict
import subprocess
import time
import signal
import concurrent.futures
import random
import tempfile

from tqdm import tqdm
from pytanque import Pytanque, PetanqueError

def extract_steps(content):
    pattern = r'\\boxed\{([\s\S]*?.)\}'
    match = re.search(pattern, content)
    if not match:
        return None
    content = match.group(1)
    instructions = re.findall(r'[^.]+?\.', content)
    return instructions

def start_pet_server(port=8765, mean_wait=10):
    """
    Starts the pet-server process and returns the process handle.
    """
    process = subprocess.Popen(["pet-server", "--port", f"{port}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # wait a bit to ensure the server is fully up before proceeding
    wait = random.randint(1, 2*mean_wait)
    time.sleep(wait)
    return process

def stop_pet_server(process):
    """
    Gracefully stops the pet-server process.
    """
    process.terminate()  # Sends SIGTERM
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()  # Force kill if not terminated
        process.wait()

# Define a custom exception for timeouts
class TimeoutError(Exception):
    pass

def timeout(seconds=5, error_message="Function call timed out"):
    """
    A decorator that raises a TimeoutError if the decorated function
    does not return within 'seconds' seconds.
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        def wrapper(*args, **kwargs):
            # Set the signal handler and a timeout alarm
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

SSR_HEADER = "From Coq Require Import ssreflect ssrfun ssrbool.\n"
def eval_tactics(thm, workspace, filepath, tactics, url="127.0.0.1", port=8765, timeout=10):
    """
    Try to solve theorem "thm" in the source file "filepath" using tactics.
    """
    with open(filepath, 'r') as file:
        content_file = file.read()
    
    new_filepath = filepath.split('.')[0] + 'aux_ssreflect.v'
    with open(new_filepath, 'w') as file:
        file.write(SSR_HEADER + content_file)
    filepath = new_filepath
    with Pytanque(url, port) as pet:
        try:
            pet.set_workspace(True, workspace)
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
                res.append(entry)

                if state.proof_finished:
                    break
            except PetanqueError as e:
                entry['status'] = "error"
                entry['message'] = e.message
                res.append(entry)
                break
        return init_goals, res

def do_thm(thms, export_path, position=0, port=8765, mean_wait=1):
    server_process = start_pet_server(port=port, mean_wait=mean_wait)
    global_step = 0
    global_len = 0

    for thm in thms:
        global_len += len(thm['outputs'])
    first_eval_tactics = timeout(seconds=60)(eval_tactics)
    second_eval_tactics = timeout(seconds=10)(eval_tactics)
    first_tactic = True
    bar = tqdm(total=global_len, position=position, desc=f"Processing {os.path.basename(export_path)}")
    done = {}
    for entry in thms:
        try:
            thm_name = entry['name']
            workspace = entry['workspace']
            filepath = entry['filepath']
            for output in entry['outputs']:
                global_step += 1
                if global_step > 800:
                    stop_pet_server(server_process)
                    server_process = start_pet_server(port=port, mean_wait=1)
                    
                    global_step = 0
                    first_tactic = True

                res = [{"status": "error", "goals": [], "message": "fail", "tactic": ""}]
                goals = []
                if not output['content']:
                    res[0]['message'] = 'No content'
                    output['goals'], output['evaluation'] = goals, res
                    continue
                
                steps = extract_steps(output['content'])
                if not steps:
                    res[0]['message'] = 'No steps'
                    output['goals'], output['evaluation'] = goals, res
                    continue

                id_steps = "\n".join(steps) + entry['name']
                if id_steps in done:
                    output['goals'], output['evaluation'] = done[id_steps]
                    bar.update()
                    continue
                try:
                    if first_tactic:
                        goals, res = first_eval_tactics(thm_name, workspace, filepath, steps, port=port)
                        first_tactic = False
                    else:
                        goals, res = second_eval_tactics(thm_name, workspace, filepath, steps, port=port)
                except TimeoutError as e:
                    # print(f"Timeout: {e}. Restarting pet-server.")
                    res[0]['message'] = 'Timeout'
                    stop_pet_server(server_process)
                    server_process = start_pet_server(port=port, mean_wait=1)
                    first_tactic = True
                output['goals'], output['evaluation'] = goals, res
                done[id_steps] = (goals, res)
                bar.update()
        except Exception as e:
            print(e)
            stop_pet_server(server_process)
            return False
    with open(export_path, 'w') as file:
        json.dump(thms, file, indent=4)
    stop_pet_server(server_process)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/benchmark_new/eval_reasoning_ablation_128_new', help='Directory containing evaluations')
    parser.add_argument('--output', default='export/benchmark_corn_hard_64/', help='Dataset')
    parser.add_argument('--max-workers', type=int, default=8)
    args = parser.parse_args()
    to_do = defaultdict(list)
    done = set()
    result = defaultdict(lambda:defaultdict(list))
    os.makedirs(args.output, exist_ok=True)

    for filename in os.listdir(args.output):
        if filename.endswith('.json'):
            done.add(filename)
        
    already_encounter = {}
    total_tasks = 0
    for root, _, files in os.walk(args.input):
        for filename in files:
            if filename.endswith('.json'):
                if filename in done:
                    continue
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as file_io:
                    data = json.load(file_io)
                
                for thm in data['thms']:
                    total_tasks += len(thm['outputs'])
                to_do[data['category']] = data['thms']
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for k, category in enumerate(to_do, start=1):
            export_path = os.path.join(args.output, category + '.json')
            futures.append(executor.submit(do_thm, to_do[category], export_path=export_path, position=k, port=8765 + k, mean_wait=3*args.max_workers))
        
        for _ in tqdm(concurrent.futures.as_completed(futures), desc="Overall progress", position=0, total=len(futures)):
            pass
    
