import json
import os
import argparse
import re

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/steps/step_1', help='Directory containing json evaluation files')
    parser.add_argument('--export', default='export/experiment/exp_0', help='Directory to export plot')
    args = parser.parse_args()

    os.makedirs(args.export, exist_ok=True)
    pattern = r'term_([0-9]+)\.json'
    term_lengths = []
    proof_lengths = []
    for root, subdirs, files in os.walk(args.input):
        for filename in files:
            match = re.match(pattern, filename)
            if match:
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as file:
                    data = json.load(file)
                
                term_lengths.append(data['term_len'])
                proof_lengths.append(data['proof_len'])
    percentiles = [50, 60, 70, 80, 85, 90, 95]

    percentile_values = [np.percentile(term_lengths, p) for p in percentiles]
    term_lengths = [length for length in term_lengths if length < percentile_values[-1]]

    # Create a scatter plot for the correlation
    plt.figure(figsize=(12, 6))

    # Histogram subplot
    plt.subplot(1, 2, 1)
    plt.hist(term_lengths, bins=50, 
            edgecolor='black', align='left', alpha=0.7)
    plt.title("Histogram of Tokenized Lambda-term Lengths")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    # Plot vertical lines for percentiles
    for p, value in zip(percentiles, percentile_values):
        plt.axvline(x=value, color='red', linestyle='--', label=f"{p}th Percentile: {value:.2f}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.legend()

    percentile_values = [np.percentile(proof_lengths, p) for p in percentiles]
    proof_lengths = [length for length in proof_lengths if length < percentile_values[-1]]

    # Histogram subplot
    plt.subplot(1, 2, 2)
    plt.hist(proof_lengths, bins=50, 
            edgecolor='black', align='left', alpha=0.7)
    plt.title("Histogram of Tokenized Proof Lengths")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    # Plot vertical lines for percentiles
    for p, value in zip(percentiles, percentile_values):
        plt.axvline(x=value, color='red', linestyle='--', label=f"{p}th Percentile: {value:.2f}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # Save the combined plot
    export_path = os.path.join(args.export, 'tokens.png')
    plt.savefig(export_path, bbox_inches='tight')
    plt.close()