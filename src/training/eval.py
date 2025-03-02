import argparse
import os
import json
import re
from collections import defaultdict
import subprocess
import time
import signal

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

def start_pet_server():
    """
    Starts the pet-server process and returns the process handle.
    """
    process = subprocess.Popen(["pet-server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # wait a bit to ensure the server is fully up before proceeding
    time.sleep(1)
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

@timeout(seconds=60)
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
                res.append(entry)
            except PetanqueError as e:
                entry['status'] = "error"
                entry['message'] = e.message
                res.append(entry)
                break
        return init_goals, res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/eval_babel', help='Directory containing evaluations')
    parser.add_argument('--input-sources', default='export/steps/sources', help='Directory containing sources files')
    parser.add_argument('--output', default='export/eval_babel.json', help='Dataset')
    args = parser.parse_args()
    to_do = []
    done = set()
    result = defaultdict(list)

    if os.path.exists(args.output):
        with open(args.output, 'r') as file:
            data = json.load(file)
        

        for thm in data:
            for entry in data[thm]:
                result[thm].append(entry)
                done.add(entry['filename'])
        
    pattern = r'term_(\S*)_sample_([0-9]+).json'
    for root, _, files in os.walk(args.input):
        for file in files:
            match = re.match(pattern, file)
            if match:
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as file_io:
                    data = json.load(file_io)
                
                data['filename'] = file
                if file in done:
                    continue
                to_do.append((data['category'] + '_' + data['name'], data['filename'], data))
    
    to_do = sorted(to_do, key=lambda x:x[0])
    server_process = start_pet_server()

    for k, (_, filename, data) in tqdm(enumerate(to_do)):
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
        try:
            _, res = eval_tactics(thm, source_path, steps)
        except TimeoutError as e:
            print(f"Timeout at step {k}: {e}. Restarting pet-server.")
            stop_pet_server(server_process)
            server_process = start_pet_server()
        if res[-1]['status'] == 'finish':
            print("SUCCESS")
        else:
            print('FAIL')
        result[thm].append({"filename": filename, "evaluation": res, "input": data})

        if k%1000 == 800:
            with open(args.output, 'w') as file:
                json.dump(result, file, indent=4)
            stop_pet_server(server_process)
            server_process = start_pet_server()

    with open(args.output, 'w') as file:
        json.dump(result, file, indent=4)
