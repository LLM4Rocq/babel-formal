import json
import os
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Any

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def draw_uniform_across_keys(
    buckets: Dict[int, List[Any]],
    n: int = 500
) -> List[Tuple[int, Any]]:
    """
    Draw up to `n` (key, item) pairs from a dict-of-lists of the form {int: [items...]},
    aiming for uniformity across keys. If some keys exhaust, it keeps going with the rest.
    Items are drawn without replacement within each key.
    """
    pool = {k: list(v) for k, v in buckets.items() if v}
    active_keys = [k for k, v in pool.items() if v]
    result: List[Tuple[int, Any]] = []

    def pop_random_from_key(k: int):
        arr = pool[k]
        i = random.randrange(len(arr))
        arr[i], arr[-1] = arr[-1], arr[i]
        return arr.pop()

    # Round-robin over (shuffled) keys, removing exhausted ones as we go
    while active_keys and len(result) < n:
        random.shuffle(active_keys)  # avoid bias within each round
        next_active = []
        for k in active_keys:
            if len(result) >= n:
                break
            if pool[k]:
                item = pop_random_from_key(k)
                result.append((k, item))
                if pool[k]:  # still has items left
                    next_active.append(k)
        active_keys = next_active

    return result

import json
import os
import argparse
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Any

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def draw_uniform_across_keys(
    buckets: Dict[int, List[Any]],
    n: int = 500
) -> List[Tuple[int, Any]]:
    """
    Draw up to `n` (key, item) pairs from a dict-of-lists of the form {int: [items...]},
    aiming for uniformity across keys. If some keys exhaust, it keeps going with the rest.
    Items are drawn without replacement within each key.
    """
    pool = {k: list(v) for k, v in buckets.items() if v}
    active_keys = [k for k, v in pool.items() if v]
    result: List[Tuple[int, Any]] = []

    def pop_random_from_key(k: int):
        arr = pool[k]
        i = random.randrange(len(arr))
        arr[i], arr[-1] = arr[-1], arr[i]
        return arr.pop()

    # Round-robin over (shuffled) keys, removing exhausted ones as we go
    while active_keys and len(result) < n:
        random.shuffle(active_keys)  # avoid bias within each round
        next_active = []
        for k in active_keys:
            if len(result) >= n:
                break
            if pool[k]:
                item = pop_random_from_key(k)
                result.append((k, item))
                if pool[k]:  # still has items left
                    next_active.append(k)
        active_keys = next_active

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="export/dataset/step_5", help="Directory with step 5 JSON files")
    parser.add_argument("--output", default="export/dataset/step_6", help="Output JSON file")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of proofs to sample")
    args = parser.parse_args()

    result = []
    for filename in tqdm(os.listdir(args.input), position=0, desc="Files remaining", disable=False):
        filepath = os.path.join(args.input, filename)
        with open(filepath, 'r') as file:
            entry = json.load(file)

        new_symbols = []
        for symbol in entry['symbols']:
            if symbol['print_text']:
                if 'theorem ' in symbol['print_text']:
                    symbol['category'] = 'theorem'
                if 'def ' in symbol['print_text']:
                    symbol['category'] = 'definition'
                if 'class ' in symbol['print_text']:
                    symbol['category'] = 'class'
                if 'inductive ' in symbol['print_text']:
                    symbol['category'] = 'inductive'
                if 'constructor ' in symbol['print_text']:
                    symbol['category'] = 'constructor'
                new_symbols.append(symbol)

        del entry['statement']
        del entry['range']
        entry['symbols'] = new_symbols
        result.append(entry)

    dict_len = defaultdict(list)
    for entry in result:
        proof_len = entry['proof'].count('\n')
        if proof_len > 3:
            dict_len[proof_len].append(entry)

    result = draw_uniform_across_keys(dict_len, args.sample_size)
    proof_len = [entry[0] for entry in result]
    result = [entry[1] for entry in result]

    if len(proof_len) >= 2:
        bins = np.histogram_bin_edges(proof_len, bins='fd')
    else:
        bins = 'auto'

    plt.figure(figsize=(6, 4))
    plt.hist(proof_len, bins=bins, edgecolor='black')
    plt.xlabel('Proof length')
    plt.ylabel('Count')
    plt.title('Histogram of Lean 4 proof script lengths')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(os.path.join(args.output, 'step_6.json'), 'w') as file:
        json.dump(result, file, indent=4)


if __name__ == "__main__":
    main()
