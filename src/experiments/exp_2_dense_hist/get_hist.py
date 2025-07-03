import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

cwd = Path(__file__).parent
prompt_path = os.path.join(cwd, 'prompt_compare')
data_path = os.path.join(cwd, 'data')
export_path = os.path.join(Path(__file__).parent, 'export/')

with open(prompt_path, 'r') as file:
    prompt_compare_template = file.read()

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

for root, subdirs, files in os.walk(data_path):
    if 'diag_v5.json' in files:
        filepath = os.path.join(root, 'diag_v5.json')
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        probs_avg = []
        probs_hidden_avg = []
        
        probs_decision = []
        probs_hidden_decision = []

        probs_completion = []
        probs_hidden_completion = []
        
        length_sample = []

        mixed_avg = data['mixed'][0][1]
        baseline_avg = data['no_reasoning'][0][1]

        mixed_decision = data['mixed'][1][1]
        baseline_decision = data['no_reasoning'][1][1]

        mixed_completion = data['mixed'][2][1]
        baseline_completion = data['no_reasoning'][2][1]

        max_val = 0.
        best_reasoning = ""
        
        y_check = []
        result_xy = []
        for entry in data:
            if '_hidden' not in entry and entry != 'mixed' and entry != 'no_reasoning':
                num_idx = entry.split('_')[1].split('.')[0]
                check_path = os.path.join(root, f'prompt_check_{num_idx}.json')
                with open(check_path, 'r') as file:
                    check_data = json.load(file)

                is_yes = '\\boxed{yes}' in check_data['content']
                is_no = '\\boxed{no}' in check_data['content']

                if is_yes or is_no:
                    y_check.append(1 if is_yes else 0)
                else:
                    y_check.append(0.5)
                reasoning_path = os.path.join(root, entry)
                with open(reasoning_path, 'r') as file:
                    reasoning_data = json.load(file)
                max_val = max(data[entry + '_hidden'][0][1], max_val)

                if max_val == data[entry + '_hidden'][0][1]:
                    best_reasoning = reasoning_data['reasoning'] + entry
                
                reasoning_len = len(reasoning_data['reasoning'].split(' '))
                length_sample.append(reasoning_len)
                prob_avg = data[entry][0][1]
                prob_hidden_avg = data[entry + '_hidden'][0][1]

                prob_decision = data[entry][1][1]
                prob_hidden_decision = data[entry + '_hidden'][1][1]

                prob_completion = data[entry][2][1]
                prob_hidden_completion = data[entry + '_hidden'][2][1]

                probs_avg.append(prob_avg)
                probs_hidden_avg.append(prob_hidden_avg)

                probs_decision.append(prob_decision)
                probs_hidden_decision.append(prob_hidden_decision)

                probs_completion.append(prob_completion)
                probs_hidden_completion.append(prob_hidden_completion)

                if is_yes:
                    result_xy.append((prob_hidden_decision, reasoning_data))
        probs_avg = np.array(probs_avg)
        probs_hidden_avg = np.array(probs_hidden_avg)

        probs_decision = np.array(probs_decision)
        probs_hidden_decision = np.array(probs_hidden_decision)

        probs_completion = np.array(probs_completion)
        probs_hidden_completion = np.array(probs_hidden_completion)

        plt.figure(figsize=(25,10))
        select_probs = probs_hidden_decision
        ax = plt.subplot(1, 4, 1)
        ax.scatter(length_sample, select_probs)
        ax.set_ylabel('Score (percentage of matching token)')
        ax.set_xlabel('Reasoning length (num words)')
        ax.set_title('Correlation score/reasoning length')

        ax = plt.subplot(1, 4, 2)
        ax.hist(select_probs, bins=10)
        ax.set_ylabel('Number of sample')
        ax.set_xlabel('Score')
        ax.set_title('Histogram of score (hidden reasoning)')

        ax = plt.subplot(1, 4, 3)
        ax.hist(length_sample, bins=10)
        ax.set_ylabel('Number of sample')
        ax.set_xlabel('Reasoning length')
        ax.set_title('Histogram of reasoning length')

        select_probs = probs_hidden_decision

        scores, checks = zip(*result_xy)

        # Convert to numpy arrays for convenience
        scores = np.array(probs_hidden_decision)
        checks = np.array(y_check)

        # 1) Define the bins you want for your scores
        #    For example, use 5 bins between the min and max of your scores.
        num_bins = 10
        bins = np.linspace(scores.min(), scores.max(), num_bins + 1)

        # 2) Digitize scores to figure out which bin each belongs to
        bin_indices = np.digitize(scores, bins)

        # 3) Calculate the mean correctness per bin
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # midpoints of each bin
        mean_correctness = []
        for i in range(1, len(bins)):
            # Get checks for all scores that fell into bin i
            in_bin = checks[bin_indices == i]
            if len(in_bin) > 0:
                mean_correctness.append(np.mean(in_bin))
            else:
                mean_correctness.append(mean_correctness[-1])

        ax = plt.subplot(1, 4, 4)
        ax.plot(bin_centers, mean_correctness, marker='o', linestyle='-', color='blue')
        ax.set_title("Probability of Correct Formatting vs. Score")
        ax.set_xlabel("Score (binned)")
        ax.set_ylabel("Mean Correctness")
        ax.set_ylim(0.,1.2)
        result_xy = sorted(result_xy)

        low_reasoning = result_xy[0][1]['reasoning']
        mid_reasoning = result_xy[len(result_xy)//2][1]['reasoning']
        high_reasoning = result_xy[-1][1]['reasoning']

        pair_reasoning = [('low_high', low_reasoning, high_reasoning), ('low_mid', low_reasoning, mid_reasoning), ('mid_high', mid_reasoning, high_reasoning)]
        for descr, reasoning_1, reasoning_2 in pair_reasoning:
            prompt = prompt_compare_template.format(reasoning_1=reasoning_1, reasoning_2=reasoning_2)

            filepath = os.path.join(root, f'compare_{descr}')
            with open(filepath, 'w') as file:
                file.write(prompt)
        
        # print("##########BEST REASONING#################")
        # print(best_reasoning)
        filename = root.split('/')[-1]
        plt.savefig(os.path.join(export_path, filename+'.png'))
        plt.close()
