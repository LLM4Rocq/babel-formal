import os
import json

from tqdm import tqdm

from src.coqpyt_ext.proof_file_mod import ProofFileMod

def get_all_v_file(folder):
    v_files = set()
    set_filename = set()
    for root, subdirs, files in os.walk(folder):
        if 'mathcomp' in root:
            for file in files:
                # we check coqpyt not in filename to avoid aux files
                if file.endswith('.v') and 'coqpyt' not in file and file not in set_filename:
                    v_files.add((os.path.join(root, file), file.split('.')[0]))
                    set_filename.add(file)
    return v_files

v_files = get_all_v_file('/home/theo/.opam/default/.opam-switch/sources/')

count_proof = 0
count_lemma = 0
count_theorem = 0

v_files_filtereed = set()
for filepath, filename in v_files:
    with open(filepath, 'r') as file:
        content = file.read()

        subcount_lemma = content.count('\nLemma')
        subcount_theorem = content.count('\nTheorem')

        if subcount_lemma + subcount_theorem > 0:
            v_files_filtereed.add((filepath, filename))
        count_lemma += content.count('\nLemma')
        count_theorem += content.count('\nTheorem')
        count_proof += content.count('\nProof.')

print(f"Lemma count: {count_lemma}\nTheorem count: {count_theorem}\nProof count: {count_proof}")
remains = set()
with open('export/below.json', 'r') as file:
    remains = set(json.load(file))

for filepath, filename in tqdm(v_files_filtereed):
    fullpath = f'export/{filename}'
    if os.path.exists(fullpath):
        if os.listdir(fullpath):
            continue
    else:
        os.mkdir(fullpath)
    with ProofFileMod(filepath, timeout=60*15) as proof_file:
        proof_file.run()
        all_terms = proof_file._extract_all_terms_v2(do_existentials=False, remains=remains)
        for k, term in enumerate(all_terms.values()):
            filepath = os.path.join(fullpath, f'term_{k}.json')
            with open(filepath, 'w') as file:
                json.dump(term, file, indent=4)