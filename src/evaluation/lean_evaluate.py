import leanclient as lc
import json
import os
import re
from tqdm import tqdm
from collections import defaultdict

from src.evaluator.factory import make_prover
from src.evaluator.prover import DatasetItem, MessageType

prover = make_prover('lean', 'dataset/')

def extract_proof(proof):
    pattern = re.compile(r'<\/think>\s*(.*?)<think>', re.DOTALL)

    blocks = pattern.findall(proof+'<think>')
    proof = []
    for block in blocks:
        if '\\box' in block:
            pattern = re.compile(r'\\box{(.*)}', re.DOTALL)
            subblock = pattern.match(block).group(1)
            proof.append(subblock.strip())
        else:
            proof.append(block.strip())
    return proof


dict_name_to_range = {}
dict_name_to_proof = {}
for source in os.listdir('dataset/json'):
    if source.endswith('.json'):
        filepath = os.path.join('dataset/json', source)
        with open(filepath, 'r') as file:
            content = json.load(file)
        
        for entry in content['items']:
            dict_name_to_range[entry['name']] = entry['lean']['lines']
            dict_name_to_proof[entry['name']] = entry['lean']['proof']

res = defaultdict(lambda:False)
for folder in tqdm(os.listdir('eval/result_lean')):
    folder_path = os.path.join('eval/result_lean', folder)
    for filename in tqdm(os.listdir(folder_path), position=0, desc="Files remaining", disable=True):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as file:
            content = json.load(file)
        source = content['source']
        name = content['name']
        line_start, line_end = dict_name_to_range[name]
        item = DatasetItem("lean", source.replace('.lean', ''), name, lines=(line_start, line_end))
        success = False
        if res[name]:
            continue
        for proof in content['outputs']:
            try:
                # print(proof['content'])
                full_proof = "\n".join(extract_proof(proof['content']))
                last_proof = extract_proof(proof['content'])[-1]
                partial_proof = "\n".join(extract_proof(proof['content'])[:-1])

                if len(partial_proof) <= len(last_proof)*1.1:
                    prover.start_thm(item)
                    message = prover.check_proof(last_proof)
                    if message.status == MessageType.FINISH:
                        success = True
                        res[name] = True
                        break
                else:
                    prover.start_thm(item)
                    message = prover.check_proof(full_proof)
                    if message.status == MessageType.FINISH:
                        success = True
                        res[name] = True
                        break
            except Exception as e:
                print(e)
                pass
        if success:
            print(name, "SUCCESS")
        else:
            print(name, "FAIL")

print(res)