import os
import json
from typing import Optional

from dataclasses import dataclass, field, asdict
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
import torch
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# -----------------------------------------------------------------------------
# Custom Evaluation Callback that Pre-tokenizes the Eval Data
# -----------------------------------------------------------------------------
# class CustomEvalCallback(transformers.TrainerCallback):
#     def __init__(self, eval_dataset, tokenizer, output_path, batch_size=8, pass_k=5):
#         """
#         Pre-tokenize the evaluation prompts to avoid tokenizing at every evaluation.
#         Generates k possible answers for each prompt and writes out a JSON file that
#         includes both the generated outputs and the original dataset entry.
#         """
#         self.tokenizer = tokenizer
#         self.output_path = output_path
#         self.batch_size = batch_size
#         self.pass_k = pass_k

#         # Store the full dataset entries for later export.
#         self.examples = list(eval_dataset)
#         # Extract raw texts (assumed stored under "text") for tokenization.
#         self.prompts = [example["text"] for example in self.examples]
#         # Pre-tokenize the prompts once. (These tensors are on CPU.)
#         self.tokenized_prompts = tokenizer(self.prompts, return_tensors="pt", padding=True, truncation=True, padding_side="left")

#     def on_step_begin(self, args, state, control, **kwargs):
#         """
#         At the beginning of a training step, generate k answers for every prompt in the eval set.
#         Each GPU gets a subset of prompts; within each GPU, prompts are batched to benefit from parallel generation.
#         """
#         # Determine distributed rank and world size.
#         if torch.distributed.is_initialized():
#             rank = torch.distributed.get_rank()
#             world_size = torch.distributed.get_world_size()
#         else:
#             rank = 0
#             world_size = 1
#         num_samples = self.tokenized_prompts["input_ids"].shape[0]
#         # Each GPU processes a subset of indices.
#         indices = list(range(rank, num_samples, world_size))
#         all_results = []
#         model = kwargs["model"]
#         model.eval()
#         pbar = None
#         if rank == 0:
#             pbar = tqdm(total=num_samples)

#         with torch.no_grad():
#             # Process assigned prompts in batches.
#             for batch_start in range(0, len(indices), self.batch_size):
#                 batch_indices = indices[batch_start: batch_start + self.batch_size]
#                 # Pad the batch.
#                 batch_inputs = {key: value[batch_indices].to(model.device)
#                     for key, value in self.tokenized_prompts.items()}
#                 # # Decode outputs. The shape is (batch_size * pass_k, sequence_length).
#                 # decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#                 # # Group the outputs for each prompt.
#                 # for i, idx in enumerate(batch_indices):
#                 #     prompt_outputs = decoded_outputs[i * self.pass_k: (i + 1) * self.pass_k]
#                 #     result_entry = {
#                 #         "dataset_entry": self.examples[idx],
#                 #         "generated_outputs": prompt_outputs
#                 #     }
#                 #     all_results.append(result_entry)
#                 if pbar:
#                     pbar.update(self.batch_size)

#         model.train()
#         # Save all results to a JSON file named uniquely with the rank and global step.
#         filepath = os.path.join(self.output_path, f'eval_rank{rank}_step{state.global_step}.json')
#         with open(filepath, 'w') as file:
#             json.dump(all_results, file, indent=4)

@dataclass
class TrainingConfig:
    block_size: int = field(default=10_000)
    dagger: bool = field(default=False)

def train():

    prompt_path = "src/training/prompts/prompt_qwen"

    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()

    with open(prompt_path, 'r') as file:
        prompt_template = file.read()

    dataset = load_dataset("json", data_files={
        'train': 'export/train.json',
        'validation': 'export/validation.json',
        'test': 'export/test.json'
    })
    dataset = dataset.map(lambda x: {
        "constants": "\n".join(x['constants']),
        "notations": "\n".join(x['notations']),
        "proof": "\n".join(x['steps'])
    })
    dataset = dataset.map(lambda x: {"text": prompt_template.format(**x)})

    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)
    
    # Set up trainer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant\n"
    # Use a token that is never used.
    tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses.
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # custom_eval_callback = CustomEvalCallback(
    #     eval_dataset=dataset['validation'],
    #     tokenizer=tokenizer,
    #     output_path=args.output_dir,
    #     batch_size=1,
    #     pass_k=16
    # )
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        data_collator=collator,
        args=args,
        # callbacks=[custom_eval_callback]
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()