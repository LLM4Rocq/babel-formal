import argparse
import os
import json
from collections import defaultdict
import random
from datetime import datetime

from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/test.json', help='Dataset')
    parser.add_argument('--workspace', default='dataset', help='Directory containing sources files')
    parser.add_argument('--model-path', default='/lustre/fsn1/projects/rech/tdm/ulu88xb/new-babel/babel-translate', help='Dataset')
    parser.add_argument('--tokenizer-path', default='/lustre/fsn1/projects/rech/tdm/ulu88xb/new-babel/babel-translate', help='Dataset')

    parser.add_argument('--prompt-path', default='config/prompts/prompt_rocq.json', help='Rocq prompt')
    parser.add_argument('--output', default='export/eval', help='Output directory')

    parser.add_argument('--k', type=int, default=16, help='Number of generation per entry')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Max output len')
    parser.add_argument('--gpus', type=int, default=4, help='Number of gpus')
    args = parser.parse_args()
    folder_name = datetime.now() .strftime("result_log_%m_%d_%H_%M_%S") + str(random.randint(0,1000))
    output_path = os.path.join(args.output, folder_name)
    os.makedirs(output_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    llm = LLM(model=args.model_path, tokenizer=args.tokenizer_path, max_num_seqs=16, max_model_len=12_000, tensor_parallel_size=args.gpus, dtype="bfloat16", gpu_memory_utilization=0.98, trust_remote_code=True)

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p)
    sampling_params.n = args.k  # This tells vLLM to generate k completions per prompt.
    
    with open(args.prompt_path, 'r') as file:
        prompt = json.load(file)

    with open(args.input, 'r') as file:
        dataset = json.load(file)
    
    dataset_lean_to_rocq = defaultdict(list)
    
    for entry in dataset:
        lean_entry = entry['lean']
        rocq_entry = entry['rocq']

        term = entry['lean']['term']
        dependencies = entry['rocq']['dependencies']
        source = rocq_entry['source']

        dataset_lean_to_rocq[source].append({
            "name": entry['lean']['name'],
            "term": term,
            "initial_goal": entry['rocq']['initial_goal'],
            "dependencies": dependencies,
            "source": source
        })

    sources = list(dataset_lean_to_rocq.keys())
    random.shuffle(sources)
    for source in sources:
        for entry in dataset_lean_to_rocq[source]:
            filepath = os.path.join(output_path, entry['name']+'.json')
            if os.path.exists(filepath):
                continue
            messages = [
                {"role": "user", "content": prompt['instruction'].format(term=entry['term'], dependencies=entry['dependencies'])}
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            initial_goal = "\n".join(entry['initial_goal'])
            prompt_text += f'<think> Okay, let\'s try to transform this proof term into a sequence of coq tactics. First let\'s write down the hypotheses, and the initial goal (after the "|-" symbol) given by the coq proof assistant:\n{initial_goal}.'
            outputs = llm.generate([prompt_text], sampling_params)
            result = []
            for output in outputs:
                for completion in output.outputs:
                    result.append(completion.text)
            
            new_entry = {"name": entry['name'], "source": source, "workspace": args.workspace, "outputs": [{"content": content} for content in result]}
            with open(filepath, 'w') as file:
                json.dump(new_entry, file, indent=4)
            