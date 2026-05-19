from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from src.common.new_benchmark import get_workspace, load_benchmark_entries
from src.evaluator.factory import make_prover
from src.evaluator.prover import DatasetItem, MessageType


_PROOF_BLOCK_RE = re.compile(r"</think>\s*(.*?)<think>", re.DOTALL)
_BOX_RE = re.compile(r"\\box{(.*)}", re.DOTALL)


def _normalize_source(source: str) -> str:
    if not source:
        return ""
    return Path(source).stem


def _extract_proof_chunks(raw_output: str) -> List[str]:
    matches = _PROOF_BLOCK_RE.findall(raw_output + "<think>")
    if not matches:
        stripped = raw_output.strip()
        return [stripped] if stripped else []

    chunks: List[str] = []
    for block in matches:
        candidate = block.strip()
        if "\\box" in candidate:
            boxed = _BOX_RE.match(candidate)
            if boxed:
                candidate = boxed.group(1).strip()
        if candidate:
            chunks.append(candidate)
    return chunks


def _candidate_proofs(raw_output: str) -> List[str]:
    chunks = _extract_proof_chunks(raw_output)
    if not chunks:
        return []

    full = "\n".join(chunks).strip()
    candidates = [full]
    last = chunks[-1].strip()
    if last and last != full:
        candidates.append(last)
    return [proof for proof in candidates if proof]


def _build_item_index(entries: List[dict], kind: str) -> Tuple[Dict[Tuple[str, str], DatasetItem], Dict[str, DatasetItem]]:
    by_key: Dict[Tuple[str, str], DatasetItem] = {}
    by_name: Dict[str, List[DatasetItem]] = defaultdict(list)

    for entry in entries:
        side = entry.get(kind)
        if not isinstance(side, dict):
            continue
        name = side.get("name") or entry.get("name")
        source = _normalize_source(side.get("source", ""))
        if not name or not source:
            continue
        name = str(name)

        if kind == "lean":
            raw_lines = side.get("lines") or [0, 0]
            lines = (int(raw_lines[0]), int(raw_lines[1]))
            item = DatasetItem("lean", source, name=name, lines=lines)
        else:
            item = DatasetItem("rocq", source, name=name, lines=(0, 0))

        key = (source, name)
        by_key[key] = item
        by_name[name].append(item)

    unique_name_map = {name: items[0] for name, items in by_name.items() if len(items) == 1}
    return by_key, unique_name_map


def _iter_result_files(results_dir: str) -> Iterable[Path]:
    root = Path(results_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"))


def _iter_output_texts(payload: dict) -> Iterable[str]:
    for output in payload.get("outputs", []):
        if isinstance(output, dict):
            yield str(output.get("content", ""))
        else:
            yield str(output)


def build_parser(kind: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="export/eval", help="Directory containing generated outputs")
    parser.add_argument("--input", default=None, help="Optional benchmark JSON. If omitted, use new_benchmark.")
    parser.add_argument("--benchmark-root", default="new_benchmark", help="Benchmark root when --input is omitted.")
    parser.add_argument("--workspace", default=None, help="Prover workspace. Defaults to new_benchmark workspace.")
    return parser


def run(kind: str) -> None:
    parser = build_parser(kind)
    args = parser.parse_args()

    entries = load_benchmark_entries(input_path=args.input, benchmark_root=args.benchmark_root)
    item_index, unique_name_map = _build_item_index(entries, kind=kind)

    workspace = args.workspace
    if workspace is None:
        workspace = str(get_workspace(kind, benchmark_root=Path(args.benchmark_root)))
    prover = make_prover(kind, workspace)

    success_by_key: Dict[Tuple[str, str], bool] = defaultdict(lambda: False)
    missing = 0
    processed = 0

    result_files = list(_iter_result_files(args.results))
    for filepath in tqdm(result_files):
        with open(filepath, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        name = str(payload.get("name", ""))
        source = _normalize_source(str(payload.get("source", "")))
        key = (source, name)
        if success_by_key[key]:
            continue

        item = item_index.get(key)
        if item is None:
            item = unique_name_map.get(name)
        if item is None:
            missing += 1
            continue

        success = False
        for output_text in _iter_output_texts(payload):
            for proof in _candidate_proofs(output_text):
                try:
                    prover.start_thm(item)
                    message = prover.check_proof(proof)
                except Exception:
                    continue
                if message.status == MessageType.FINISH:
                    success = True
                    break
            if success:
                break

        success_by_key[key] = success
        processed += 1

    success_count = sum(1 for ok in success_by_key.values() if ok)
    attempted = len(success_by_key)
    total = len(item_index)
    print(f"Kind: {kind}")
    print(f"Benchmark entries: {total}")
    print(f"Attempted entries: {attempted}")
    print(f"Successful entries: {success_count}")
    print(f"Success rate on attempted: {success_count / attempted:.2%}" if attempted else "Success rate on attempted: n/a")
    print(f"Missing benchmark mapping in outputs: {missing}")
    print(f"Processed result files: {processed} / {len(result_files)}")

    if hasattr(prover, "close"):
        try:
            prover.close()
        except Exception:
            pass


def main(kind: str) -> None:
    if kind not in {"lean", "rocq"}:
        raise ValueError(f"Unknown kind: {kind}")
    run(kind)
