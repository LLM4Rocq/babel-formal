import os
import json
from pathlib import Path
import re
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/evaluation/result', help='Directory containing json evaluation files')
    parser.add_argument('--export', default='export/experiment/exp_7', help='Directory to export plot')
    parser.add_argument('--source', default='export/benchmark_corn_hard.json')
    args = parser.parse_args()

    os.makedirs(args.export, exist_ok=True)

    pattern = r'(bench|eval)_(.*)\.json'
    # Dictionary to hold, for each model, a mapping of proof length to counts of successes and totals.
    model_accuracy_by_length = {}

    with open(args.source, 'r') as file:
        benchmark = json.load(file)
    
    name_to_proof_len = {}
    steps_lengths = []
    for entry in benchmark:
        name_to_proof_len[entry['name']] = len([s for s in entry['steps'] if 'Proof.' not in s and 'Qed.' not in s])
    for file in sorted(os.listdir(args.input)):
        match = re.match(pattern, file)
        if not match:
            continue
        model_name = match.group(2)
        model_accuracy_by_length[model_name] = {}
        filepath = os.path.join(args.input, file)
        with open(filepath, 'r') as f:
            data_eval = json.load(f)
        
        for thm in data_eval:
            for entry in data_eval[thm]:
                proof_length = len(entry['evaluation'])
                steps_lengths.append(proof_length)
                if 'No steps' in entry['evaluation'][-1]['message'] or 'No content' in entry['evaluation'][-1]['message']:
                    continue
                if proof_length not in model_accuracy_by_length[model_name]:
                    model_accuracy_by_length[model_name][proof_length] = {'success': 0, 'total': 0}
                
                eval_entry = entry['evaluation'][-1]

                model_accuracy_by_length[model_name][proof_length]['total'] += 1
                if eval_entry['status'] == 'finish':
                    model_accuracy_by_length[model_name][proof_length]['success'] += 1

    # Plot accuracy vs. proof length for each model.
    plt.figure(figsize=(10, 6))
    for model, length_data in model_accuracy_by_length.items():
        # Get sorted proof lengths.
        lengths = sorted([L for L in length_data.keys() if length_data[L]['total'] > 4])
        # Compute accuracy (in percentage) for each proof length.
        accuracies = [100 * length_data[L]['success'] / length_data[L]['total'] for L in lengths]
        plt.plot(lengths, accuracies, marker='o', label=model)
    
        plt.ylim((-0.5, 102))
        plt.xlabel('Generated proof Length (number of steps)')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy vs Generated Proof Length')
        plt.legend()
        plt.grid(True)
        
        export_path = os.path.join(args.export, f'{model}_accuracy_by_length.png')
        plt.savefig(export_path, bbox_inches='tight')
        plt.close()

        # Create a scatter plot for the correlation
        plt.figure(figsize=(12, 6))

        plt.hist(steps_lengths, bins=max(steps_lengths),
                edgecolor='black', align='left', alpha=0.7)
        plt.title("Histogram of Proof Lengths")
        plt.xlabel("Number of steps")
        # plt.xlim(0, 20)
        plt.ylabel("Frequency")
        plt.legend()

        # Save the combined plot
        export_path = os.path.join(args.export, f'{model}_steps.png')
        plt.savefig(export_path, bbox_inches='tight')
        plt.close()