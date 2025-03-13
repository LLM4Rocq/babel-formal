import os
import json
from functools import partial
import gc
import argparse

import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from transformers.optimization import get_cosine_schedule_with_warmup

#from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='export/')
    parser.add_argument("--export", type=str, default='ckpt/')
    parser.add_argument("--prompt-path", type=str, default='src/training/prompts/prompt.json')
    # parser.add_argument(
    #     "--model-dir", type=str, default="arnir0/Tiny-LLM"
    # )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lr", type=float, default=1e-05, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--empty-cache", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers to process dataset")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--lr-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-dir", type=str)
    args = parser.parse_args()
    return args

args = parse_args()
accelerator = Accelerator()

writer = None
if accelerator.is_main_process:
    writer = SummaryWriter(log_dir=args.log_dir)
DEVICE = accelerator.device

torch.manual_seed(53)
torch.cuda.manual_seed(53)


def pad_sequences(sequences, pad_value, pad_first=False):
    """
    Pads a list of sequences (lists of ints) to the same length using pad_value.
    """
    max_length = max(len(seq) for seq in sequences)
    if pad_first:
        return torch.tensor([[pad_value] * (max_length - len(seq)) + seq for seq in sequences])
    else:
        return torch.tensor([seq + [pad_value] * (max_length - len(seq)) for seq in sequences])

def list_of_dict_to_dict(lst):
    if not lst:
        return {}
    result = {key:[] for key in lst[0].keys()}
    for dct in lst:
        for key in dct:
            result[key].append(dct[key])
    return result

def merge_and_pad_entries(entries, pad_value, pad_first=False):
    merge_entries = list_of_dict_to_dict(entries)
    keys = ['input_ids', 'attention_mask', 'labels']
    result = {key: pad_sequences(merge_entries[key], pad_value, pad_first=pad_first) for key in keys}
    return result


def train_loop(model, tokenizer, train_dataloader, eval_dataloader, optimizer, scheduler, args):
    model.train()
    loop = tqdm(total=int(len(train_dataloader)*args.num_epochs/args.gradient_accumulation_steps), disable=not accelerator.is_main_process)
    global_step = 0
    accumulation_steps = args.gradient_accumulation_steps
    avg_loss = torch.tensor(0., device=DEVICE)
    for epoch in range(args.num_epochs):
        for step, inputs in enumerate(train_dataloader):
            # I was unable to leverage accelerator accumulate
            if args.empty_cache:
                gc.collect()
                torch.cuda.empty_cache()
                accelerator.free_memory()
            
            if (step + 1) % accumulation_steps == 0:
                out = model(**inputs)
                accelerator.backward(out.loss/accumulation_steps)
                avg_loss += out.loss.detach()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                reduced_loss = accelerator.reduce(avg_loss,reduction="mean")/ accumulation_steps
                if accelerator.is_main_process:
                    writer.add_scalar("loss_train", reduced_loss , global_step)
                    print(f"Step: {global_step}, Loss: {reduced_loss}, lr: {optimizer.param_groups[0]['lr']}.")
                    loop.update(1)
                avg_loss = torch.tensor(0., device=DEVICE)
            else:
                with accelerator.no_sync(model):
                    out = model(**inputs)
                    accelerator.backward(out.loss/accumulation_steps)
                    avg_loss += out.loss.detach()               
        folder_path = os.path.join(args.export, f"checkpoint-epoch-{epoch}")
        model.save_pretrained(
                        folder_path,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        state_dict=accelerator.get_state_dict(model),
                    )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(folder_path)
        
    return model

def eval_loop(model, tokenizer, eval_dataloader, filename):
    model.eval()
    outputs_list = []
    with torch.no_grad():
        for inputs in eval_dataloader:
            del inputs['target']
            out = model.generate(**inputs, do_sample=True, temperature=0.5, num_return_sequences=3, max_length=10_000)
            out = accelerator.pad_across_processes(out, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
            out = accelerator.gather(out)
            outputs_list += tokenizer.batch_decode(out).tolist()
    if accelerator.is_main_process:
        with open(filename, "w") as f:
            json.dump(outputs_list, f, indent=4)

def only_keep_columns(dataset, columns):
    for key in dataset:
        cols_to_remove = dataset[key].column_names
        [cols_to_remove.remove(column) for column in columns]
        dataset[key] = dataset[key].remove_columns(cols_to_remove)

def preprocess_dataset(tokenizer, entry):
    input_ids_list_before = tokenizer(entry['before'], add_special_tokens=False)
    input_ids_list_after = tokenizer(entry['after'], add_special_tokens=False)
    input_ids_list_sep = tokenizer(entry['sep'], add_special_tokens=False)

    num_example = len(input_ids_list_before['input_ids'])
    input_ids_list = []
    labels_list = []
    attn_mask_list = []
    for i in range(num_example):
        before_ids = input_ids_list_before['input_ids'][i]
        sep_ids = input_ids_list_sep['input_ids'][i]
        after_ids = input_ids_list_after['input_ids'][i]

        input_ids_list.append(before_ids + sep_ids + after_ids)
        labels_list.append([-100] * (len(before_ids) + len(sep_ids)) +after_ids)
        attn_mask_list.append([1]*len(input_ids_list[-1]))

        # assert len(input_ids_list[-1]) == len(labels_list[-1]) and len(labels_list[-1])== len(attn_mask_list[-1])
    batch = {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attn_mask_list
    }
    return batch

def check_alignement(tokenizer, token_ids, sep, labels):
    sep_ids = tokenizer(sep, add_special_tokens=False)['input_ids']
    for i in range(len(token_ids) - len(sep_ids) + 1):
        if token_ids[i] == sep_ids[0]:
            if token_ids[i:i+len(sep_ids)] == sep_ids:
                for j in range(i + len(sep_ids)-1):
                    assert labels[j] == -100, "labels misaligned with respect to inputs_ids (sooner !=100 than expected)"
                for j in range(i+len(sep_ids)-1, len(token_ids)):
                    assert labels[j] != -100, "labels misaligned with respect to inputs_ids (=100 for too long)"
                return True
    raise Exception('Issue')

def main(args):
    with open(args.prompt_path, 'r') as file:
        prompt = json.load(file)
    # Initialize Datasets

    # Initialize Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )
    # tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
    # if not tokenizer.pad_token_id:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset("json", data_files={
        'train': os.path.join(args.data_path, 'train.json'),
        'validation': os.path.join(args.data_path, 'validation.json'),
        'test': os.path.join(args.data_path, 'test.json')
    })

    dataset = dataset.map(lambda x: {
        "constants": "\n".join(x['constants']),
        "notations": "\n".join(x['notations']),
        "proof": "\n".join(x['steps'])
    })
    dataset = dataset.map(lambda x: {"sep": prompt["sep"], "before": prompt['beg'] + prompt['text_before'].format(**x), "after": prompt['text_after'].format(**x) + prompt['end']})
    dataset = dataset.map(partial(preprocess_dataset, tokenizer), batched=True, batch_size=100)
    only_keep_columns(dataset, ['attention_mask', 'labels', 'input_ids', 'name', 'category'])
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        shuffle=True,
        collate_fn=lambda x:merge_and_pad_entries(x, tokenizer.pad_token_id, pad_first=False)
    )

    eval_dataloader = torch.utils.data.DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=lambda x:merge_and_pad_entries(x, tokenizer.pad_token_id, pad_first=True)
    )
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if not model.config.pad_token_id:
        model.config.pad_token_id = model.config.eos_token_id
    # Initialize Optimizer and Criterion
    model.gradient_checkpointing_enable({"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)

    num_training_steps = int(len(train_dataloader) *args.num_epochs/args.gradient_accumulation_steps)
    lr_warmup_iters = int(num_training_steps * args.lr_warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=lr_warmup_iters, num_training_steps=num_training_steps)

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    model = train_loop(model, tokenizer, train_dataloader, eval_dataloader, optimizer, lr_scheduler, args)


if __name__ == "__main__":
    main(args)