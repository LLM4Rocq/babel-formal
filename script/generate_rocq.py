import argparse
import os
import json
from collections import defaultdict
import concurrent.futures
import random
from typing import List

from tqdm import tqdm

from src.evaluator.rocq_prover import DatasetItem, RocqProver
from src.llm.openai_instruct import OpenAIInstructLLM
from src.agent.rocq import RocqAgent, AgentStatus


def exec(model_name: str, item: DatasetItem, output_path: str, workspace: str, max_retry=5, max_depth=32):
    llm = OpenAIInstructLLM(model_name, generation_parameters=
    {
        "max_tokens":8192,
        "temperature": 0.7,
        "top_p": 0.95,
        "extra_body": {"skip_special_tokens": False}
    })
    prover = RocqProver(workspace)
    agent = RocqAgent(llm, prover, max_retry=max_retry, max_depth=max_depth)

    output = {}
    status, instructions = agent.whole_proof_generation(item)
    output['state'] = str(agent.state)
    output['logs'] = agent.logs
    output['instructions'] = instructions

    if status == AgentStatus.FINISH:
        output_path += "_SUCCESS.json"
    else:
        output_path += "_FAIL.json"
    with open(output_path, 'w') as file:
        json.dump(output, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='benchmark/test_lean_to_rocq.json', help='Dataset')
    parser.add_argument('--workspace', default='/lustre/fsn1/projects/rech/tdm/ulu88xb/babel-formal/dataset/version/rocq/', help='Directory containing sources files')
    parser.add_argument('--model-path', default='/lustre/fsn1/projects/rech/tdm/ulu88xb/babel-formal/babel-translate', help='Dataset')
    parser.add_argument('--tokenizer-path', default='/lustre/fsn1/projects/rech/tdm/ulu88xb/babel-formal/babel-translate', help='Dataset')

    parser.add_argument('--output', default='export/eval', help='Output directory')

    parser.add_argument('--max-workers', type=int, default=32, help='Max number of concurrent workers')

    parser.add_argument('--max-retry', type=int, default=2, help='Max number of retry/block/run')
    parser.add_argument('--max-depth', type=int, default=16, help='Max depth of generated proof')
    parser.add_argument('--pass-k', type=int, default=128, help='Number of generation per entry')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Max output len')

    parser.add_argument('--gpus', type=int, default=4, help='Number of gpus')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    with open(args.input, 'r') as file:
        dataset = json.load(file)
    
    dataset_lean_to_rocq = defaultdict(list)
    
    for entry in dataset:
        source = entry['source']
        dataset_lean_to_rocq[source].append({
            "name": entry['name'],
            "term": entry['term'],
            "dependencies": entry['dependencies'],
            "source": entry['source']
        })

    to_do:List[DatasetItem] = []
    for source in tqdm(dataset_lean_to_rocq):
        for entry in dataset_lean_to_rocq[source]:
            name = entry['name']
            term = entry['term']
            dependencies = entry['dependencies']
            item = DatasetItem("rocq", source, term=term, name=name, dependencies=dependencies)
            to_do.append(item)
    
    random.shuffle(to_do)
    for item in to_do:
        futures = []
        name = item.name
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            for i in range(args.pass_k):
                output_path = os.path.join(args.output, name + f'_{i}')
                futures.append(executor.submit(exec, args.model_path, item, output_path, args.workspace, max_retry=args.max_retry, max_depth=args.max_depth))
            for _ in tqdm(concurrent.futures.as_completed(futures), desc="Pass@k", position=1, total=len(futures)):
                pass

