import leanclient as lc
import json
import os
import re
from tqdm import tqdm
from collections import defaultdict

from src.evaluator.factory import make_prover
from src.evaluator.prover import DatasetItem, MessageType

prover = make_prover('rocq', 'dataset/repo/rocq')

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


res = defaultdict(lambda:False)
for folder in tqdm(os.listdir('eval/result_rocq')):
    folder_path = os.path.join('eval/result_rocq', folder)
    for filename in tqdm(os.listdir(folder_path), position=0, desc="Files remaining", disable=True):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as file:
            content = json.load(file)
        source = content['source'].replace('dataset/repo/rocq/', '').replace('.v', '')
        name = content['name']

        if res[name]:
            continue
        line_start, line_end = 0, 0
        item = DatasetItem("rocq", source, name, lines=(line_start, line_end))
        success = False
        for proof in content['outputs']:
            try:
                if len(extract_proof(proof['content'])) < 7:
                    continue
                proof_extract = "\n".join(extract_proof(proof['content']))
                prover.start_thm(item)
                
                message = prover.check_proof(proof_extract)
                if message.status == MessageType.FINISH:
                    prover.run_tac("Qed.")
                    print(source)
                    print(proof['content'])
                    print(proof_extract)
                    # exit()
                    success = True
                    res[name] = True
                    break
            except Exception as e:
                print(e)
                print("IGNORE")
        if success:
            print(name, "SUCCESS")
        else:
            print(name, "FAIL")

print(res)