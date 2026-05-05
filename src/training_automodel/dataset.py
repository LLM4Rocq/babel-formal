"""Dataset adapters for Babel training with NeMo AutoModel."""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from datasets import VerificationMode, load_dataset

from nemo_automodel.components.datasets.lazy_mapped_dataset import LazyMappedDataset
from nemo_automodel.components.datasets.llm.formatting_utils import _add_pad_token

logger = logging.getLogger(__name__)

_SPLIT_SLICE_RE = re.compile(r"^(\w+)\[(\d*):(\d*)\]$")


def _parse_split_slice(split: Optional[str]) -> tuple[Optional[str], Optional[slice]]:
    if split is None:
        return split, None
    match = _SPLIT_SLICE_RE.match(split)
    if not match:
        return split, None

    base = match.group(1)
    start = int(match.group(2)) if match.group(2) else None
    end = int(match.group(3)) if match.group(3) else None
    return base, slice(start, end)


def _looks_like_hf_id(value: str) -> bool:
    return "/" in value and not Path(value).exists()


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "train" in payload and isinstance(payload["train"], list):
            return payload["train"]
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        raise ValueError(
            f"Unsupported dataset JSON structure in {path}. Expected a list or a dict containing `train` or `data`."
        )

    raise ValueError(f"Unsupported dataset JSON payload in {path}: {type(payload)}")


def _slice_rows(rows: List[Dict[str, Any]], split: Optional[str]) -> List[Dict[str, Any]]:
    if split is None:
        return rows

    base, sl = _parse_split_slice(split)
    if base not in (None, "train"):
        raise ValueError(f"Only `train` split is supported for local JSON datasets, got: {split}")
    if sl is None:
        return rows

    return rows[sl]


def _load_rows(path_or_dataset_id: Union[str, Sequence[str]], split: Optional[str], shuffle_seed: Optional[int]) -> List[Dict[str, Any]]:
    if isinstance(path_or_dataset_id, str) and _looks_like_hf_id(path_or_dataset_id):
        base_split, sl = _parse_split_slice(split)
        dataset = load_dataset(
            path_or_dataset_id,
            split=base_split or "train",
            streaming=False,
            verification_mode=VerificationMode.NO_CHECKS,
        )
        if shuffle_seed is not None:
            dataset = dataset.shuffle(seed=shuffle_seed)
        if sl is not None:
            dataset = dataset.select(range(*sl.indices(len(dataset))))
        return [dataset[i] for i in range(len(dataset))]

    rows: List[Dict[str, Any]] = []
    paths: Sequence[str] = [path_or_dataset_id] if isinstance(path_or_dataset_id, str) else path_or_dataset_id
    for item in paths:
        rows.extend(_read_json_or_jsonl(Path(item)))

    rows = _slice_rows(rows, split)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    return rows


def _build_messages(example: Dict[str, Any], prompt_lean: Dict[str, str], prompt_rocq: Dict[str, str]) -> List[Dict[str, str]]:
    language = str(example.get("language", "")).strip().lower()
    if language == "lean":
        prompt = prompt_lean
    elif language == "rocq":
        prompt = prompt_rocq
    else:
        raise ValueError(f"Unsupported language `{language}` in dataset example.")

    term = str(example.get("term", "")).strip()
    dependencies = str(example.get("dependencies", "")).strip()
    output = str(example.get("output", "")).strip()

    if not term:
        raise ValueError("Missing `term` in dataset example.")
    if not output:
        raise ValueError("Missing `output` in dataset example.")

    user_prompt = prompt["instruction"].format(term=term, dependencies=dependencies)
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": output},
    ]


def make_babel_reasoning_dataset(
    tokenizer,
    path_or_dataset_id: Union[str, Sequence[str]] = "training_set/lean_rocq.json",
    split: Optional[str] = "train",
    prompt_lean_filepath: str = "config/prompts/prompt_lean.json",
    prompt_rocq_filepath: str = "config/prompts/prompt_rocq.json",
    seq_length: Optional[int] = 32768,
    padding: Union[str, bool] = "do_not_pad",
    truncation: Union[str, bool] = "longest_first",
    answer_only_loss_mask: bool = True,
    mask_reasoning_content: bool = False,
    shuffle_seed: Optional[int] = 1111,
    limit_dataset_samples: Optional[int] = None,
    cache_size: Optional[int] = 10000,
):
    """Build a Babel dataset formatted for AutoModel causal LM SFT.

    Expected raw sample fields: `language`, `term`, `dependencies`, `output`.
    This reproduces the legacy NeMo supervision path:
    1) render user prompt with `add_generation_prompt=True`
    2) append raw assistant output tokens + EOS
    3) compute shifted labels where prompt/generation-prefix tokens are masked
       (`-100`) and only assistant output + EOS are supervised.
    """

    if tokenizer is None:
        raise ValueError("Tokenizer is required")

    rows = _load_rows(path_or_dataset_id=path_or_dataset_id, split=split, shuffle_seed=shuffle_seed)
    if limit_dataset_samples is not None:
        rows = rows[:limit_dataset_samples]

    logger.info("Loaded %d raw samples for split=%s from %s", len(rows), split, path_or_dataset_id)

    with open(prompt_lean_filepath, "r", encoding="utf-8") as handle:
        prompt_lean = json.load(handle)
    with open(prompt_rocq_filepath, "r", encoding="utf-8") as handle:
        prompt_rocq = json.load(handle)

    eos_token_id = getattr(tokenizer, "eos_token_id", 0)
    pad_token_id = _add_pad_token(tokenizer) or eos_token_id

    def _format_row(example: Dict[str, Any]) -> Dict[str, List[int]]:
        # Parameters kept for API compatibility with generic datasets.
        _ = (seq_length, padding, truncation, answer_only_loss_mask, mask_reasoning_content)

        messages = _build_messages(example=example, prompt_lean=prompt_lean, prompt_rocq=prompt_rocq)
        user_message = [messages[0]]
        assistant_output = messages[1]["content"]

        prompt_text = tokenizer.apply_chat_template(
            user_message,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Keep tokenizer defaults intentionally: this mirrors legacy NeMo preprocessing.
        prompt_ids = tokenizer(prompt_text)["input_ids"]
        assistant_ids = tokenizer(assistant_output)["input_ids"]

        full_ids = prompt_ids + assistant_ids + [eos_token_id]
        ignore_idx = [0] * len(prompt_ids) + [1] * (len(assistant_ids) + 1)

        input_ids = full_ids[:-1]
        labels = [tok if keep == 1 else -100 for tok, keep in zip(full_ids[1:], ignore_idx[1:])]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "___PAD_TOKEN_IDS___": {
                "input_ids": pad_token_id,
                "labels": -100,
                "attention_mask": 0,
            },
        }

    return LazyMappedDataset(rows, _format_row, cache_size=cache_size)
