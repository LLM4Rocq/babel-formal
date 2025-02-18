import os
import argparse
import json
import re



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/step_5/', help='Input dataset path')
    parser.add_argument('--output', default='export/final.json', help='Output dataset filepath')
    parser.add_argument('--max_workers', default=100, type=int, help='Max number of concurrent workers')
    parser.add_argument('--mean_delay', default=10, type=int, help='Mean delay before a request is send: use this parameter to load balance')
    args = parser.parse_args()
    
    dataset = {"train": []}
    pattern = r'(term_[0-9]+).json'
    for root, _, files in os.walk(args.output):
        for file in files:
            match = re.match(pattern, file)
            if match:
                term_name = match.group(1)
                filepath = os.path.join(root,file)

                with open(filepath, 'r') as file:
                    data = json.load(file)
                
                data['origin'] = os.path.join(root, file.split('.')[0])
                dataset["train"].append(data)

    with open(args.output, 'w') as file:
        json.dump(dataset, file)




