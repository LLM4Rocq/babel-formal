import os
import argparse
from collections import deque
import sys
import random
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


def _scores_vector(res, n):
    """
    Normalize bm25s retrieve() output into a 1D float32 vector of length n,
    aligned with corpus order, with -inf for non-returned entries.
    Supports:
      - objects with .indices and .scores (top-k)
      - objects with .scores only (aligned, may be 2-D)
      - raw arrays/lists
    """
    vec = np.full(n, -np.inf, dtype=np.float32)

    # Case A: top-k style result: has indices + scores
    if hasattr(res, "indices") and hasattr(res, "scores"):
        inds = np.asarray(res.indices)
        scs = np.asarray(res.scores, dtype=np.float32)
        # Flatten possible (1, m) shapes
        inds = inds.reshape(-1)
        scs = scs.reshape(-1)
        vec[inds] = scs
        return vec

    # Case B: aligned scores only (may be (n,), (1,n), or nested list)
    if hasattr(res, "scores"):
        scs = np.asarray(res.scores, dtype=np.float32)
    else:
        scs = np.asarray(res, dtype=np.float32)

    scs = scs.squeeze()  # (1, n) -> (n,)
    if scs.ndim != 1:
        raise ValueError(f"Unexpected scores shape: {scs.shape}")
    if scs.shape[0] != n:
        # If not length n, we can't align—treat as top-k without indices (rare)
        raise ValueError(f"Scores length {scs.shape[0]} != corpus size {n}")

    return scs


def select_diverse_documents(documents, entries, k, seed_index=None, neighbors="full"):
    n = len(documents)
    if n == 0:
        return []
    k = min(k, n)

    retriever = bm25s.BM25(corpus=documents)
    # If your bm25s needs explicit tokens for indexing, swap these two lines:
    # tokens_for_index = [bm25s.tokenize(d) for d in documents]
    # retriever.index(tokens_for_index)
    retriever.index(documents)

    tokens = [bm25s.tokenize(d) for d in documents]

    if seed_index is None:
        seed_index = random.randrange(n)

    max_sim = np.zeros(n, dtype=np.float32)

    def update_with(idx):
        if neighbors == "full":
            res = retriever.retrieve(tokens[idx], k=n)
        else:
            m = int(neighbors)
            res = retriever.retrieve(tokens[idx], k=min(m, n))

        # Normalize to aligned vector and update
        vec = _scores_vector(res, n)
        np.maximum(max_sim, vec, out=max_sim)
        max_sim[idx] = np.inf  # don't reselect same doc

    update_with(seed_index)

    selected = [seed_index]
    with tqdm(total=k, desc="Selecting") as progress:
        for _ in range(k):
            next_idx = int(np.argmin(max_sim))
            selected.append(next_idx)
            update_with(next_idx)
            progress.update(1)

    return [entries[i] for i in selected]

def round_robin_draw(buckets: dict[str, list[str]], k: int):
    """
    buckets: dict mapping key -> list of strings
    k: number of items to draw (without replacement)
    """

    # Copy & shuffle items within each bucket; filter out empties
    per_key = {k: lst[:] for k, lst in buckets.items() if lst}
    for lst in per_key.values():
        random.shuffle(lst)

    # Queue of keys that still have items; shuffle start order for fairness
    q = deque(per_key.keys())
    keys = list(q)
    random.shuffle(keys)
    q = deque(keys)

    result = []
    while q and len(result) < k:
        key = q.popleft()
        item = per_key[key].pop()   # pop from shuffled list
        result.append(item)
        if per_key[key]:            # still has items? put key back at end
            q.append(key)
        # if empty, that key drops out automatically

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='export/lean_dataset/step_3')
    parser.add_argument("--input", default='export/lean_dataset/step_2/step_2.json')

    parser.add_argument('--num-documents', default=1_300, help='Maximum number of final documents')
    parser.add_argument('--trim-documents', default=100_000, help='Maximum number of final documents')
    parser.add_argument('--min-proof-len', default=4)
    parser.add_argument('--max-proof-len', default=25)

    args = parser.parse_args()
    filepaths = {}
    
    with open(args.input, 'r') as file:
        content = json.load(file)

    tot_len = 0
    for filepath in content:
        new_entry = []
        for entry in content[filepath]:
            entry['source'] = filepath.replace('_', '/')
            proof = entry['proof']
            proof_len = len(proof.split('\n'))

            if args.min_proof_len <= proof_len <= args.max_proof_len:
                new_entry.append(entry)

        content[filepath] = new_entry

    entries = round_robin_draw(content, args.trim_documents)
    documents = [entry['lean_extract']['statement'] + ' ' + entry['proof'] for entry in entries]
    filtered_entries = select_diverse_documents(documents, entries, args.num_documents)

    with open(os.path.join(args.output, 'step_3.json'), 'w') as file:
        json.dump(filtered_entries, file, indent=4)