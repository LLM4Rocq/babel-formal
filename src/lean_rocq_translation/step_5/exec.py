import json
import os
import argparse
from tqdm import tqdm

from src.evaluator.factory import make_prover
from src.evaluator.prover import DatasetItem, MessageType


def sub2_line_span(text: str, subtext_1: str, subtext_2: str):
    combo = subtext_1 + subtext_2
    i = text.find(combo)
    if i == -1:
        raise ValueError("subtext_1 + subtext_2 not found consecutively")
    start_idx = i + len(subtext_1)
    end_idx_excl = start_idx + len(subtext_2)
    start_line = text.count("\n", 0, start_idx)
    end_line = text.count("\n", 0, max(end_idx_excl - 1, 0))
    return start_line, end_line

def preprocess_proof(proof: str):
    new_proof = ""
    prefix = ""
    lines = proof.splitlines()
    if not lines:
        return prefix, new_proof
    if lines[0] == 'by':
        prefix = lines[0]
        lines.pop(0)
    for line in lines:
        if line and (line[0] != ' ' and line[0] != '\t'):
            break
        new_proof += line + '\n'
    return prefix, new_proof.rstrip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-path", default='/home/_/Documents/github/mathlib4')
    parser.add_argument("--input", default='export/dataset/step_4')
    parser.add_argument("--output", default='export/dataset/step_5')
    args = parser.parse_args()

    project_path = args.project_path
    prover = make_prover('lean', project_path)
    ignore_count = 0
    for filename in tqdm(os.listdir(args.input), position=0, desc="Files remaining", disable=False):
        filepath = os.path.join(args.input, filename)
        with open(filepath, 'r') as file:
            content = json.load(file)
        for source in content:
            source_path = os.path.join(project_path, source+'.lean')
            for entry in tqdm(content[source], position=1, desc="Theorems remaining", disable=True):
                name = entry['name']
                header = entry['header']
                proof = entry['proof'].rstrip()
                filename = filename[:-len('.json')]
                export_filepath = os.path.join(args.output, filename + '#' + name)
                if os.path.exists(export_filepath):
                    continue
                prefix, proof = preprocess_proof(proof)

                header += ' ' + prefix + '\n'

                with open(source_path, 'r') as file:
                    source_content = file.read()

                if header+proof not in source_content or not proof.strip():
                    ignore_count += 1
                    continue
                try:
                    line_start, line_end = sub2_line_span(source_content, header, proof)
                    item = DatasetItem("lean", source, name, lines=(line_start, line_end+1))
                    prover.start_thm(item)
                    last_state = prover.check_proof(proof)
                    if last_state.status != MessageType.FINISH:
                        print(f"Incomplete proof at {item}, {last_state}")
                        continue
                    entry['source'] = source
                    entry['header'] = header
                    entry['proof'] = proof
                    with open(export_filepath, 'w') as file:
                        json.dump(entry, file, indent=4)
                except Exception as e:
                    print(e)
    print(ignore_count)

if __name__ == "__main__":
    main()


