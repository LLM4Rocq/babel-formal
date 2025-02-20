import os
import argparse
import json
from collections import defaultdict
import sys
import re
import shutil

# to avoid issue with json recursion
sys.setrecursionlimit(10_000) 

from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/step_0/', help='Output path previous step')
    parser.add_argument('--output', default='export/step_1/', help='Output path')
    parser.add_argument('--tokenizer', default='deepseek-ai/DeepSeek-Prover-V1.5-Base', help='HF path tokenizer')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    shutil.copytree(args.input, args.output, dirs_exist_ok=True)
    entries = defaultdict(dict)
    pattern = r'(term_[0-9]+).json'
    to_do = []
    for root, subdirs, files in os.walk(args.output):
        for filename in files:
            match = re.match(pattern, filename)
            if match:
                filepath = os.path.join(root, filename)
                to_do.append(filepath)
    
    for filepath in tqdm(to_do):
        with open(filepath, 'r') as file:
            entry = json.load(file)
        term_str = entry['term']
        tactics_str = "\n".join(entry['steps'])
        tokens_tactics = tokenizer(tactics_str, return_tensors="pt", truncation=False)["input_ids"]
        tokens_term = tokenizer(term_str, return_tensors="pt", truncation=False)["input_ids"]

        entry['term_len'] = tokens_term.size(1)
        entry['proof_len'] = tokens_tactics.size(1)
        
        with open(filepath, 'w') as file:
            json.dump(entry, file, indent=4)