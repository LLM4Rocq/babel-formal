import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

cwd = Path(__file__).parent
data_path = os.path.join(cwd, 'data')
export_path = os.path.join(Path(__file__).parent, 'export/')
prompt_solve_path = os.path.join(Path(__file__).parent, 'prompt_solve')
prompt_solve_term_path = os.path.join(Path(__file__).parent, 'prompt_solve_term')

with open(prompt_solve_path, 'r') as file:
    prompt_solve_template = file.read()

with open(prompt_solve_term_path, 'r') as file:
    prompt_solve_term_template = file.read()

for root, subdirs, files in os.walk(data_path):
    if 'data.json' in files:
        with open(os.path.join(root, 'data.json'), 'r') as file:
            data = json.load(file)
        
        with open(os.path.join(root, 'prompt_solve'), 'w') as file:
            prompt_solve = prompt_solve_template.format(lemma=data['proposition'][0], constants="\n".join(data['constant']))
            file.write(prompt_solve)
        
        with open(os.path.join(root, 'prompt_solve_term'), 'w') as file:
            prompt_solve = prompt_solve_term_template.format(term=data['term'][0], constants="\n".join(data['constant']))
            file.write(prompt_solve)