import os
import json
from pathlib import Path
import re
import argparse
import random
from math import log2, floor
from copy import deepcopy
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


def wilson(p, n, z = 1.96):
    p = p/100
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = sqrt((p*(1 - p) + z*z / (4*n)) / n)

    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return (lower_bound*100, upper_bound*100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/evaluation/result', help='Directory containing json evaluation files')
    parser.add_argument('--export', default='export/experiment/exp_5', help='Directory to export plot')
    args = parser.parse_args()

    os.makedirs(args.export, exist_ok=True)


    pattern = r'(bench|eval)_(.*)\.json'
    k_parameter = 0
    scores = {}
    all_result = []
    for file in sorted(os.listdir(args.input)):
        match = re.match(pattern, file)
        if not match:
            continue
        model_name = match.group(2)


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
        y_low = []
        y_high = []

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
            percentage = valid_thm/num_thm*100
            low, high = wilson(percentage, num_thm)
            y_result.append(percentage)
            y_low.append(low)
            y_high.append(high)

        all_result.append((model_name, lengths, y_result, y_low, y_high))

    
    plt.figure(figsize=(10, 6))
    for model_name, x, y, y_low, y_high in all_result:
        plt.plot(x, y, label=model_name)
        plt.fill_between(x, y_low, y_high ,alpha=0.3)

    plt.title(f'Scaling law (test-time compute)')
    plt.ylabel('Mean accuracy')
    plt.xlabel('k parameter')
    plt.legend()
    export_path = os.path.join(args.export, f'scaling.png')
    plt.savefig(export_path, bbox_inches='tight')
    plt.close()