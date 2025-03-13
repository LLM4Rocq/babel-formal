import os
import json
from pathlib import Path
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/evaluation/result', help='Directory containing json evaluation files')
    parser.add_argument('--export', default='export/experiment/exp_4', help='Directory to export plot')
    args = parser.parse_args()

    os.makedirs(args.export, exist_ok=True)


    pattern = r'(bench|eval)_(.*)\.json'
    k_parameter = 0
    scores = {}
    for file in sorted(os.listdir(args.input)):
        match = re.match(pattern, file)
        if not match:
            continue
        model_name = match.group(1)
        scores[model_name] = {"fail": 0, "success": 0, "total": 0}
        filepath = os.path.join(args.input, file)
        with open(filepath, 'r') as file:
            data_eval = json.load(file)
        
        for thm in data_eval:
            success = False
            scores[model_name]['total'] += 1
            k_parameter = max(k_parameter, len(data_eval[thm]))
            for entry in data_eval[thm]:
                entry = entry['evaluation']
                res = entry[-1]['status']

                if res == 'finish':
                    success = True
            if success:
                scores[model_name]['success'] += 1
            else:
                scores[model_name]['fail'] += 1

        distrib = []
        for thm in data_eval:
            score = 0
            for entry in data_eval[thm]:
                entry = entry['evaluation'][-1]
                res = entry['status']
                if res == 'finish':
                    score += 1
            distrib.append(score)
        
        plt.figure(figsize=(10, 6))
        plt.hist(distrib, bins=np.arange(-0.5, k_parameter+1.5, 1))
        plt.title(f'Histogram of score of {model_name}')
        export_path = os.path.join(args.export, f'hist_{model_name}.png')
        plt.savefig(export_path, bbox_inches='tight')
        plt.close()
        


    # Compute success percentage
    model_names = list(scores.keys())
    success_percentages = [(scores[m]['success'] / scores[m]['total']) * 100 if scores[m]['total'] > 0 else 0 for m in model_names]
    
    x = np.arange(len(model_names))  # the label locations
    width = 0.2  # the width of the bars
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, success_percentages, width, color='green')
    
    plt.ylabel('Success Percentage')
    plt.xlabel('Models')
    plt.xticks(ticks=x, labels=model_names, rotation=45, ha='right')
    plt.title(f'Success Rate for All Models (pass@k)')
    plt.ylim(0, 100)
    # Add text labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.1f}%', ha='center', va='bottom')
    
    # Save the combined plot
    export_path = os.path.join(args.export, 'all_models.png')
    plt.savefig(export_path, bbox_inches='tight')
    plt.close()