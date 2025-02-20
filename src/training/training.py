import os
import json
from typing import Optional

from dataclasses import dataclass, field, asdict
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
import torch

# -----------------------------------------------------------------------------
# Custom Evaluation Callback that Pre-tokenizes the Eval Data
# -----------------------------------------------------------------------------
class CustomEvalCallback(transformers.TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, output_path, batch_size=8):
        """
        Pre-tokenize the evaluation prompts to avoid tokenizing at every evaluation.
        """
        self.tokenizer = tokenizer
        self.output_path = output_path
        self.batch_size = batch_size

        # Extract raw texts from the evaluation dataset.
        self.raw_texts = [example.get("text", "") for example in eval_dataset][:8]
        # Construct prompts based on your instruction template.
        self.prompts = [f"<|im_start|>user {text}" for text in self.raw_texts]
        # Pre-tokenize the prompts once. (These tensors are on CPU.)
        self.tokenized_prompts = tokenizer(
            self.prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        # In multi-GPU/distributed setups, only have the main process perform file I/O.
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return control

        model = kwargs["model"]
        device = args.device if hasattr(args, "device") else "cpu"
        model.eval()

        # Move the pre-tokenized inputs to the correct device.
        tokenized_prompts = {key: val.to(device) for key, val in self.tokenized_prompts.items()}

        results = []
        num_samples = len(self.raw_texts)
        # Process the prompts in batches.
        for i in range(0, num_samples, self.batch_size):
            batch_slice = slice(i, i + self.batch_size)
            batch_inputs = {key: val[batch_slice] for key, val in tokenized_prompts.items()}
            with torch.no_grad():
                outputs = model.generate(**batch_inputs, max_length=512)
            # Decode the outputs in a batch.
            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # Store the raw input along with the generated output.
            for raw, generated in zip(self.raw_texts[i:i + self.batch_size], decoded_outputs):
                results.append({
                    "input": raw,
                    "generated_text": generated
                })
        # Save the results to a JSON file.
        filepath = os.path.join(self.output_path, f'eval_{state.global_step}.json')
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Custom evaluation results saved to {filepath}")

        return control


def train():
    # parsing input
    model_name = "sbintuitions/tiny-lm"
    file_path = "simplescaling/s1K_tokenized"


    training_args = trl.SFTConfig(
        max_seq_length=512,
        output_dir="export/training",
        report_to="tensorboard",
        logging_dir="export/tensorboard",
        dataset_text_field='text',
        logging_strategy="steps",
        logging_steps=10
    )
    dataset = load_dataset(file_path)
    dataset = dataset.map(lambda x: {"text":"<|im_start|>user Hello! <|im_start|>assistant\n Hello!"})
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

    
    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)

    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant\n"
    # Use a token that is never used
    tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    eval_dataset = dataset['test'] if 'test' in dataset else dataset['train']
    # Instantiate the custom evaluation callback.
    custom_eval_callback = CustomEvalCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        output_path=training_args.output_dir,
        batch_size=8
    )

    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        data_collator=collator,
        args=training_args,
        callbacks=[custom_eval_callback]
    )

    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()