import json
import os

from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Load the tokenizer for Llama (replace with the correct model identifier or path)
model_name = "deepseek-ai/DeepSeek-Prover-V1.5-Base"  # E.g., "meta-llama/Llama-2-7b-hf" or your local model path
tokenizer = AutoTokenizer.from_pretrained(model_name)

entries = {}
for root, subdirs, files in os.walk('export'):
    for filename in files:
        if filename.endswith('.json'):
            filepath = os.path.join(root, filename)
            with open(filepath, 'r') as file:
                entries[root.split('/')[-1] + '_' + filename.split('.')[0]] = json.load(file)
# with open('test.json', 'r') as file:
#     entries = json.load(file)
# Calculate the number of tokens for each string

tokens_term_lengths = []
proof_lengths = []
for entry in tqdm(entries.values()):
    if entry['term']:
        term_str = entry['term'][0]
        tactics_str = "\n".join(entry['steps'])
        tokens_tactics = tokenizer(tactics_str, return_tensors="pt", truncation=False)["input_ids"]
        tokens_term = tokenizer(term_str, return_tensors="pt", truncation=False)["input_ids"]
        tokens_term_lengths.append(tokens_term.size(1))
        proof_lengths.append(tokens_tactics.size(1))
    else:
        print(f'Issue with {entry}')

# # Calculate percentiles
len_tok = len(tokens_term_lengths)
tokens_term_lengths_trunc = sorted(tokens_term_lengths)
tokens_term_lengths_trunc = tokens_term_lengths_trunc[:int(len_tok*95/100)]

print(f"Number of removed lambda-term: {len_tok - len(tokens_term_lengths_trunc)}")

len_tok = len(tokens_term_lengths)
proof_lengths_trunc = sorted(proof_lengths)
proof_lengths_trunc = proof_lengths_trunc[:int(len_tok*95/100)]

print(f"Number of removed proof: {len_tok - len(proof_lengths_trunc)}")

percentiles = [50, 60, 70, 80, 85, 90, 95]

percentile_values = [np.percentile(tokens_term_lengths_trunc, p) for p in percentiles]


# Create a scatter plot for the correlation
plt.figure(figsize=(12, 6))

# Histogram subplot
plt.subplot(1, 3, 1)
plt.hist(tokens_term_lengths_trunc, bins=20, 
         edgecolor='black', align='left', alpha=0.7)
plt.title("Histogram of Tokenized Lambda-term Lengths")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
# Plot vertical lines for percentiles
for p, value in zip(percentiles, percentile_values):
    plt.axvline(x=value, color='red', linestyle='--', label=f"{p}th Percentile: {value:.2f}")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

percentile_values = [np.percentile(proof_lengths_trunc, p) for p in percentiles]

# Histogram subplot
plt.subplot(1, 3, 2)
plt.hist(proof_lengths_trunc, bins=20, 
         edgecolor='black', align='left', alpha=0.7)
plt.title("Histogram of Tokenized Proof Lengths")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
# Plot vertical lines for percentiles
for p, value in zip(percentiles, percentile_values):
    plt.axvline(x=value, color='red', linestyle='--', label=f"{p}th Percentile: {value:.2f}")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

max_proof_len = max(proof_lengths_trunc)
max_term_len = max(tokens_term_lengths)
proof_lengths_mut_trunc = []
tokens_term_lengths_mut_trunc = []

for proof_len, term_len in zip(proof_lengths, tokens_term_lengths):
    if proof_len< max_proof_len and term_len < max_term_len:
        proof_lengths_mut_trunc.append(proof_len)
        tokens_term_lengths_mut_trunc.append(term_len)

# Scatter plot for correlation
plt.subplot(1, 3, 3)
plt.scatter(proof_lengths_mut_trunc, tokens_term_lengths_mut_trunc, color='blue', alpha=0.7)
plt.title(f"Correlation between Lambda-term length and proof length.")
plt.xlabel("Proof length")
plt.ylabel("term Length")
plt.grid(alpha=0.5)

plt.tight_layout()

plt.show()