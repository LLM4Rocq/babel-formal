import os
import argparse
from collections import defaultdict
import json

"""
Second step: match extracted Lean terms with proof dataset and save aligned results
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lean-terms', default='src/extract_term/proofs-dump/proofs_by_theorem/', help='Mathlib terms')
    parser.add_argument('--input', default='export/lean_dataset/step_1/', help='Mathlib proofs')
    parser.add_argument('--num-documents', default=2_000, help='Maximum number of final documents')
    parser.add_argument('--trim-documents', default=200_000, help='Maximum number of final documents')
    parser.add_argument("--output", default='export/lean_dataset/step_2')

    args = parser.parse_args()

    proof_term_pattern = '_proof_term.txt'
    statement_pattern = '_statement.txt'
    
    os.makedirs(args.output, exist_ok=True)
    term_dict = {}
    for root, _, files in os.walk(args.lean_terms):
        for filename in files:
            filepath = os.path.join(root, filename)

            origin = root.split('/')[-1]
            with open(filepath, 'r') as file:
                content = file.read()
            if origin not in term_dict:
                term_dict[origin] = {}

            if filename.endswith(proof_term_pattern):
                name = filename[:-len(proof_term_pattern)]
                if name not in term_dict[origin]:
                    term_dict[origin][name] = {}
                term_dict[origin][name]['term'] = content
            if filename.endswith(statement_pattern):
                name = filename[:-len(statement_pattern)]
                if name not in term_dict[origin]:
                    term_dict[origin][name] = {}
                term_dict[origin][name]['statement'] = content

    dict_coincide=defaultdict(list)
    count_ok=0
    count_notfound=0
    count_notunique=0
    for root, subdirs, files in os.walk(args.input):
        for filename in files:
            filepath = os.path.join(root, filename)
            with open(filepath, 'r') as file:
                content = json.load(file)

            for filepath in content:
                if filename not in term_dict:
                    print(f"IGNORE: {filename} not in lean terms")
                    continue
                for entry in content[filepath]:
                    name = entry['name']
                    found = ""
                    unique = True

                    for term_name in term_dict[filename]:
                        if term_name.endswith(name):
                            if found:
                                unique = False
                            found = term_name
                    if not found:
                        # print(f"{name} not found in {filename}")
                        count_notfound += 1
                    elif not unique:
                        # print(f"{name} not unique in {filename}")
                        count_notunique += 1
                    else:
                        dict_coincide[filename].append(entry)
                        entry['fqn'] = found
                        entry['lean_extract'] = term_dict[filename][found]
                        count_ok += 1
    
    with open(os.path.join(args.output, 'step_2.json'), 'w') as file:
        json.dump(dict(dict_coincide), file, indent=4)

