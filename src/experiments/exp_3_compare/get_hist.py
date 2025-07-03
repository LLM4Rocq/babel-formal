import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

cwd = Path(__file__).parent
data_path = os.path.join(cwd, 'data/compare.json')
export_path = os.path.join(Path(__file__).parent, 'export/')

with open(data_path, 'r') as file:
    data = json.load(file)

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

class_nums = {"low":0, "medium":0, "high":0}
for value in data.values():
    fst, snd, thd = value

    class_nums[fst] += 1
    class_nums[snd] += 2
    class_nums[thd] += 3

width = 0.4  # the width of the bars
space = 0.8
step_size = 4
plt.figure(figsize=(20,10))

ax = plt.subplot(1, 4, 2)
rects = ax.bar(0.5*width, np.array([class_nums['low']]).round(decimals=1), width, label='low class', color='midnightblue')
ax.bar_label(rects, padding=3)

rects = ax.bar(1.5*width + space, np.array([class_nums['medium']]).round(decimals=1), width, label='medium class', color='mediumblue')
ax.bar_label(rects, padding=3)

rects = ax.bar(2.5*width+ 2*space, np.array([class_nums['high']]).round(decimals=1), width, label='high class', color='cornflowerblue')
ax.bar_label(rects, padding=3)

ax.set_ylabel('Total score (according to GPT-o3-mini) for each class')
ax.legend(loc='upper left')
plt.savefig(os.path.join(export_path, 'comparison.png'))
plt.close()