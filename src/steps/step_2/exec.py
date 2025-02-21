import os
import argparse
from collections import defaultdict
import sys
import re
import shutil
import json

# to avoid issue with json recursion
sys.setrecursionlimit(10_000) 

import numpy as np
import bm25s
from tqdm import tqdm
"""
Third step: Filter previous dataset based on terms length and number of steps in proof, then select a diverse subset using BM25.
"""


def delete_empty_folders(root):
    """
    Deletes folder if empty
    """
    deleted = set()
    for current_dir, subdirs, files in os.walk(root, topdown=False):
        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break
    
        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)

def select_diverse_documents(documents, filepaths, k):
    """
    Extracts subset of diverse documents using BM25.
    """
    # Not efficient, but enough for the moment
    retriever = bm25s.BM25(corpus=documents)
    retriever.index(bm25s.tokenize(documents))
    similarity_matrix = np.zeros((len(documents), len(documents)))
    
    for i, doc in tqdm(enumerate(documents)):
        scores = retriever.get_scores([doc])
        similarity_matrix[i, :] = scores

    selected_indices = [0]  # Start with the first document
    while len(selected_indices) < k:
        min_similarities = []
        
        for i in range(len(documents)):
            if i not in selected_indices:
                min_sim = min(similarity_matrix[i, selected_indices])
                min_similarities.append((i, min_sim))
        next_doc = min(min_similarities, key=lambda x: x[1])[0]
        selected_indices.append(next_doc)
    
    return set([filepaths[i] for i in selected_indices])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/step_1/', help='Output path previous step')
    parser.add_argument('--output', default='export/step_2/', help='New output path')
    parser.add_argument('--num-documents', default=1_000, help='Maximum number of final documents')
    parser.add_argument('--max-num-tokens', default=3_750, help='Maximum number of tokens in term')
    parser.add_argument('--min-number-instructions', default=3, help='Minimum number of steps in a proof')
    parser.add_argument('--max-number-instructions', default=7, help='Maximum number of steps in a proof')

    args = parser.parse_args()
    shutil.copytree(args.input, args.output, dirs_exist_ok=True)
    entries = defaultdict(dict)
    pattern = r'(term_[0-9]+).json'
    documents = []
    filepaths = []
    for root, subdirs, files in os.walk(args.output):
        for filename in files:
            match = re.match(pattern, filename)
            if match:
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as file:
                    entry = json.load(file)
                
                len_steps = len(entry['steps'])

                if 7 < len_steps or len_steps < 3 or entry['term_len']  > args.max_num_tokens:
                    continue

                document = entry['proposition'] + '\n' + "\n".join(entry['steps'])
                documents.append(document)
                filepaths.append(filepath)
    
    filepaths_to_keep = select_diverse_documents(documents, filepaths, args.num_documents)

    for root, subdirs, files in os.walk(args.output):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filepath not in filepaths_to_keep:
                os.remove(filepath)
    delete_empty_folders(args.output)
