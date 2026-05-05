#!/usr/bin/env python3
"""Quick inspector for Babel AutoModel dataset masking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _load_tokenizer(model_name: str):
    # Prefer NeMoAutoTokenizer to mirror training; fallback to HF tokenizer.
    try:
        from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer

        return NeMoAutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _format_piece(tokenizer: Any, token_id: int) -> str:
    piece = tokenizer.convert_ids_to_tokens([token_id])[0]
    # Make whitespace/newline explicit in terminal output.
    return piece.replace("\n", "\\n").replace("\t", "\\t")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect token masking for Babel training samples.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dataset-path", type=str, default="training_set/lean_rocq.json")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--index", type=int, default=0, help="Sample index in the split.")
    parser.add_argument("--num-samples", type=int, default=1, help="How many consecutive samples to print.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max token rows to print per sample.")
    args = parser.parse_args()

    # Allow running from babel-formal even when AutoModel is only present as sibling checkout.
    local_automodel = Path(__file__).resolve().parents[3] / "Automodel"
    if local_automodel.exists():
        sys.path.insert(0, str(local_automodel))

    from dataset import make_babel_reasoning_dataset

    tokenizer = _load_tokenizer(args.model_name)
    dataset = make_babel_reasoning_dataset(
        tokenizer=tokenizer,
        path_or_dataset_id=args.dataset_path,
        split=args.split,
    )

    for offset in range(args.num_samples):
        idx = args.index + offset
        sample = dataset[idx]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attention_mask = sample["attention_mask"]

        print("=" * 100)
        print(f"sample_index={idx}")
        print(f"input_tokens={len(input_ids)}")
        print(f"supervised_tokens={sum(1 for x in labels if x != -100)}")
        print(f"masked_tokens={sum(1 for x in labels if x == -100)}")
        print("-" * 100)
        print("pos\tattn\tin_id\tlabel\tmask\tpiece")

        max_rows = min(len(input_ids), args.max_tokens)
        for pos in range(max_rows):
            in_id = input_ids[pos]
            label = labels[pos]
            attn = attention_mask[pos]
            mask = "S" if label != -100 else "M"
            piece = _format_piece(tokenizer, in_id)
            print(f"{pos}\t{attn}\t{in_id}\t{label}\t{mask}\t{piece}")

        if max_rows < len(input_ids):
            print(f"... truncated {len(input_ids) - max_rows} token rows")

        supervised_label_ids = [tok for tok in labels if tok != -100]
        print("-" * 100)
        print("decoded_full_input:")
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        print("-" * 100)
        print("decoded_supervised_target_tokens_only:")
        print(tokenizer.decode(supervised_label_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
