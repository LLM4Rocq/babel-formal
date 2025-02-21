import os
import argparse

import sys
# to avoid issue with json recursion
sys.setrecursionlimit(10_000) 

from tqdm import tqdm
from src.coqpyt_extension.proof_file_mod import ProofFileMod
from coqpyt.lsp.structs import ResponseError

"""
First step: compile mathcomp to extract terms, notations, constants, steps etc.
"""

def get_all_source_files(folderpath:str, filter:str='mathcomp'):
    """
    Extract all sources files (.v files) in sub directories of "folderpath" containing the variable "filter", by default corresponds to mathcomp library.
    """
    v_files = set()
    set_filename = set()
    stats = {}
    tot_count_lemma = 0
    tot_count_theorem = 0
    tot_count_proof = 0
    for root, _, files in os.walk(folderpath):
        if filter in root:
            for file in files:
                filepath = os.path.join(root, file)
                # few checks: not already done, not an auxiliary file, and is a source file
                if file in set_filename or 'coqpyt' in file or not file.endswith('.v'):
                    continue
                with open(filepath, 'r') as file_io:
                    content = file_io.read()
                count_lemma = content.count('\nLemma')
                count_theorem = content.count('\nTheorem')
                count_proof = content.count('\nProof.')

                set_filename.add(file)

                # check if file contains proof(s)
                if count_proof == 0:
                    continue
                
                stats[file] = {'count_lemma': count_lemma, 'count_theorem': count_theorem, 'count_proof': count_proof}
                tot_count_lemma += count_lemma
                tot_count_theorem += count_theorem
                tot_count_proof += count_proof
                v_files.add((os.path.join(root, file), file.split('.')[0]))
    
    stats_tot = {'count_lemma': tot_count_lemma, 'count_theorem': tot_count_theorem, 'count_proof': tot_count_proof}
    return v_files, stats, stats_tot

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Mathcomp path')
    parser.add_argument('--output', default='export/step_0/', help='Output dataset path')
    parser.add_argument('--timeout', default=1*60, type=int, help='Coqpyt timeout')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of worker')
    parser.add_argument('--idx-worker', default=0, type=int, help='Index of current worker')

    args = parser.parse_args()
    
    v_files, _, stats_tot = get_all_source_files('/home/theo/.opam/default/.opam-switch/sources/')
    print(f"Lemma count: {stats_tot['count_lemma']}\nTheorem count: {stats_tot['count_theorem']}\nProof count: {stats_tot['count_proof']}")
    remains = set()

    
    v_files = list(v_files)
    v_files_filtered = []
    v_files = sorted(v_files)
    for filepath, filename in v_files:
        fullpath = os.path.join(args.output, filename)
        if os.path.exists(os.path.join(fullpath, 'finish')):
            continue
        v_files_filtered.append((filepath, filename))

    v_files_filtered = list(chunks(v_files_filtered, len(v_files_filtered)//args.num_workers))[args.idx_worker]
    for filepath, filename in tqdm(v_files_filtered):
        print(filepath)
        fullpath = os.path.join(args.output, filename)
        os.makedirs(fullpath, exist_ok=True)
        if os.path.exists(os.path.join(fullpath, 'finish')):
            continue
        
        for _ in range(100):
            try:
                with ProofFileMod(filepath, timeout=args.timeout) as proof_file:
                    proof_file.run()
                    proof_file.extract_one_by_one(fullpath)
                with open(os.path.join(fullpath, 'finish'), 'w'):
                    pass
                break
            except FileNotFoundError as e:
                print(e)
            except ResponseError as e:
                print(e)