import os
import argparse
import concurrent.futures
import sys
import json
# to avoid issue with json recursion
sys.setrecursionlimit(10_000) 


from tqdm import tqdm
from src.coqpyt_extension.proof_file_mod import ProofFileMod
from coqpyt.lsp.structs import ResponseError
from coqpyt.coq.exceptions import NotationNotFoundException

"""
First step: compile mathcomp to extract terms, notations, constants, steps etc.
"""

def get_all_source_files(folderpath:str):
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
        for file in files:
            filepath = os.path.join(root, file)
            # few checks: not already done, not an auxiliary file, and is a source file
            if file in set_filename or 'coqpyt' in file or not file.endswith('.v'):
                continue
            try:
                with open(filepath, 'r', errors='ignore') as file_io:
                    content = file_io.read()
            except FileNotFoundError:
                continue
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

def execute_file(filepath, export_path, repository, workspace, timeout, num_retry=100):
    for _ in range(num_retry):
        try:
            with ProofFileMod(filepath, workspace=os.path.abspath(workspace), timeout=timeout) as proof_file:
                proof_file.run()
                metadata = {'repository': repository, 'workspace': workspace, 'filepath': filepath}
                proof_file.extract_one_by_one(export_path, metadata=metadata)
            with open(os.path.join(export_path, 'finish'), 'w'):
                pass
            break
        except FileNotFoundError as e:
            print(e)
        except ResponseError as e:
            print(e)
        except NotationNotFoundException as e:
            print(e)
        except Exception as e:
            print(filepath)
            print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repositories', default='export/repositories_github_rocq/', help='Repositories')
    parser.add_argument('--output', default='export/steps/step_0_github/', help='Output dataset path')
    parser.add_argument('--timeout', default=1*60, type=int, help='Coqpyt timeout')
    parser.add_argument('--max-workers', default=8, type=int, help='Number of workers')
    args = parser.parse_args()
    
    v_files_repo = {}

    for folder in os.listdir(args.repositories):
        workspace = os.path.join(args.repositories, folder)
        v_files, _, stats_tot = get_all_source_files(workspace)
        print(f"Repository: {folder}\n Lemma count: {stats_tot['count_lemma']}\nTheorem count: {stats_tot['count_theorem']}\nProof count: {stats_tot['count_proof']}")
        remains = set()
        v_files = list(v_files)
        v_files_filtered = []
        v_files = sorted(v_files)
        for filepath, filename in v_files:
            export_path = os.path.join(args.output, folder, filename)
            if os.path.exists(os.path.join(export_path, 'finish')):
                continue
            v_files_filtered.append((workspace, filepath, export_path, filename))
        v_files_repo[folder] = v_files_filtered

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for repository in v_files_repo:
            for workspace, filepath, export_path, filename in v_files_repo[repository]:
                os.makedirs(export_path, exist_ok=True)
                futures.append(executor.submit(execute_file, filepath, export_path, repository, workspace, timeout=args.timeout))
            
        for _ in tqdm(concurrent.futures.as_completed(futures), desc="Overall progress", position=0, total=len(futures)):
            pass


