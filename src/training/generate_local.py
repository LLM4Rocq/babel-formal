import argparse
import os
import json

from transformers import AutoTokenizer
from tqdm import tqdm
# Import vLLM classes
from vllm import LLM, SamplingParams

from src.training.dataset import load_and_process

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dataset', default='export/', help='Dataset')
    parser.add_argument('--input-sources', default='export/steps/sources', help='Directory containing sources files')
    parser.add_argument('--model-path', default='/lustre/fsn1/projects/rech/mmr/ulu88xb/babel/checkpoint-epoch-3', help='Dataset')
    parser.add_argument('--tokenizer-path', default='/lustre/fsn1/projects/rech/mmr/ulu88xb/models/Qwen-32B', help='Dataset')

    parser.add_argument('--prompt-path', default='src/training/prompts/prompt.json', help='Dataset')
    parser.add_argument('--output', default='export/eval', help='Output directory')
    parser.add_argument('--dataset-entry', default='validation', help='Entry to use in the dataset')
    parser.add_argument('--k', default=64, help='Number of generation per entry')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    args.tokenizer_path = MODEL_NAME
    args.model_path = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    dataset = load_and_process(tokenizer, args.input_dataset, args.prompt_path)
    llm = LLM(model=args.model_path, tokenizer=args.tokenizer_path, max_model_len=12_000, dtype="bfloat16", gpu_memory_utilization=0.98, trust_remote_code=True)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=4096, top_p=0.95)
    sampling_params.n = args.k  # This tells vLLM to generate k completions per prompt.

    for entry in tqdm(dataset[args.dataset_entry]):
        filepath = os.path.join(args.output, entry['name']+'.json')
        if os.path.exists(filepath):
            continue
        # Convert token IDs back into text.
        prompt_text = tokenizer.decode(entry['input_ids'], skip_special_tokens=True)
        # Generate output completions.
        outputs = llm.generate([prompt_text], sampling_params)
        # Each output in "outputs" is a RequestOutput object containing a list of completions.
        result = []
        for output in outputs:
            for completion in output.outputs:
                result.append(completion.text)
        
        with open(filepath, 'w') as file:
            json.dump({"name": entry['name'], "category": entry['category'], outputs: result}, file, indent=4)