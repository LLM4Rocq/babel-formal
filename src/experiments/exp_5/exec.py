import os
import json
from pathlib import Path
import re
import argparse
import random
from math import log2, floor
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/evaluation/result', help='Directory containing json evaluation files')
    parser.add_argument('--export', default='export/experiment/exp_5', help='Directory to export plot')
    args = parser.parse_args()

    os.makedirs(args.export, exist_ok=True)


    pattern = r'bench_(.*)\.json'
    k_parameter = 0
    scores = {}
    all_result = []
    for file in sorted(os.listdir(args.input)):
        match = re.match(pattern, file)
        if not match:
            continue
        model_name = match.group(1)


        filepath = os.path.join(args.input, file)
        with open(filepath, 'r') as file:
            data_ref = json.load(file)
        
        for thm in data_ref:
            random.shuffle(data_ref[thm])
        all_lengths = [len(data_ref[thm]) for thm in data_ref]
        max_len = max(all_lengths)
        exponent = floor(log2(max_len))
        lengths = [2**i for i in range(exponent+1)]

        if lengths[-1] < max_len:
            lengths.append(max_len)

        y_result = []
        for length in lengths:
            num_thm = 0
            valid_thm = 0
            for thm in data_ref:
                success = False
                num_thm += 1
                for entry in data_ref[thm][:length]:
                    entry = entry['evaluation']
                    res = entry[-1]['status']

                    if res == 'finish':
                        success = True
                
                if success:
                    valid_thm += 1
            y_result.append(valid_thm/num_thm)
        all_result.append((model_name, lengths, y_result))

    
    plt.figure(figsize=(10, 6))
    for model_name, x, y in all_result:
        plt.plot(x, y, label=model_name)
    plt.title(f'Scaling law (test-time compute)')
    plt.ylabel('Mean accuracy')
    plt.xlabel('k parameter')
    plt.legend()
    export_path = os.path.join(args.export, f'scaling.png')
    plt.savefig(export_path, bbox_inches='tight')
    plt.close()