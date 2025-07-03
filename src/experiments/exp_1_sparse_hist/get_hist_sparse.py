import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

data_path = os.path.join(Path(__file__).parent, 'data')
export_path = os.path.join(Path(__file__).parent, 'export/')

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
        
        mixed_avg = data['mixed'][0][1]
        baseline_avg = data['no_reasoning'][0][1]

        mixed_decision = data['mixed'][1][1]
        baseline_decision = data['no_reasoning'][1][1]

        mixed_completion = data['mixed'][2][1]
        baseline_completion = data['no_reasoning'][2][1]
        for entry in data:
            if '_hidden' not in entry and entry != 'mixed' and entry != 'no_reasoning':
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
        
        probs_avg = np.array(probs_avg)
        probs_hidden_avg = np.array(probs_hidden_avg)

        probs_decision = np.array(probs_decision)
        probs_hidden_decision = np.array(probs_hidden_decision)

        probs_completion = np.array(probs_completion)
        probs_hidden_completion = np.array(probs_hidden_completion)

        width = 0.4  # the width of the bars
        space = 0.8
        step_size = 4

        x = np.arange(len(probs_avg)*step_size, step=step_size)  # the label locations

        baseline_x = np.array([-step_size])
        fig, ax = plt.subplots(figsize=(20, 10))
        
        rects = ax.bar(baseline_x+0.5*width, np.array([baseline_avg]).round(decimals=1), width, label='No reasoning (average)', color='midnightblue')
        ax.bar_label(rects, padding=3)

        rects = ax.bar(baseline_x+1.5*width, np.array([baseline_decision]).round(decimals=1), width, label='No reasoning (decision)', color='mediumblue')
        ax.bar_label(rects, padding=3)

        rects = ax.bar(baseline_x+2.5*width, np.array([baseline_completion]).round(decimals=1), width, label='No reasoning (completion)', color='cornflowerblue')
        ax.bar_label(rects, padding=3)

        rects = ax.bar(baseline_x+space+2.5*width, np.array([mixed_avg]).round(decimals=1), width, label='Mixed (average)', color='black')
        ax.bar_label(rects, padding=3)

        rects = ax.bar(baseline_x+space+3.5*width, np.array([mixed_decision]).round(decimals=1), width, label='Mixed (decision)', color='dimgrey')
        ax.bar_label(rects, padding=3)

        rects = ax.bar(baseline_x+space+4.5*width, np.array([mixed_completion]).round(decimals=1), width, label='Mixed (completion)', color='silver')
        ax.bar_label(rects, padding=3)
        
        
        rects = ax.bar(x+space+2.5*width, probs_avg.round(decimals=1), width, label='Full reasoning (average)', color='darkred')
        ax.bar_label(rects, padding=3)

        rects = ax.bar(x+space+3.5*width, probs_decision.round(decimals=1), width, label='Full reasoning (decision)', color='red')
        ax.bar_label(rects, padding=3)

        rects = ax.bar(x+space+4.5*width, probs_completion.round(decimals=1), width, label='Full reasoning (completion)', color='salmon')
        ax.bar_label(rects, padding=3)

        rects = ax.bar(x+0.5*width, probs_hidden_avg.round(decimals=1), width, label='Obfuscated (average)', color='darkolivegreen')
        ax.bar_label(rects, padding=3)

        rects = ax.bar(x+1.5*width, probs_hidden_decision.round(decimals=1), width, label='Obfuscated (decision)', color='olivedrab')
        ax.bar_label(rects, padding=3)

        rects = ax.bar(x+2.5*width, probs_hidden_completion.round(decimals=1), width, label='Obfuscated (completion)', color="yellowgreen")
        ax.bar_label(rects, padding=3)
        

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Mean score per token')
        ax.set_title(root.split('/')[-1])
        ax.set_xticks(list(baseline_x + step_size/2. - 0.7) + list(x + step_size/2. - 0.7), ['baseline'] + [f'sample_{k}' for k in range(len(probs_avg))])
        ax.legend(loc='upper left', ncols=4)
        ax.set_ylim(0, 100)

        filename = root.split('/')[-1]
        plt.savefig(os.path.join(export_path, filename+'.png'))
        plt.close()