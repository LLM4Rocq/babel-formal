from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

from src.common.new_benchmark import load_benchmark_entries

from .extract_proof_terms import BenchmarkProofTermExtractionJob
from .utils import load_examples


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _example_to_row(example) -> dict:
    return {
        "example_id": example.example_id,
        "direction": example.direction,
        "theorem_name": example.theorem_name,
        "source_stem": example.source_stem,
        "source_term": example.source_term,
        "dependencies": example.dependencies,
        "target_statement": example.target_statement,
        "target_initial_goals": example.target_initial_goals,
        "target_lines": list(example.target_lines) if example.target_lines else None,
    }


def _shard(rows: List[dict], num_shards: int) -> List[List[dict]]:
    if num_shards <= 1:
        return [rows]
    buckets: List[List[dict]] = [[] for _ in range(num_shards)]
    for idx, row in enumerate(rows):
        buckets[idx % num_shards].append(row)
    return buckets


def _write_shards(direction_dir: Path, rows: List[dict], num_shards: int) -> None:
    shards = _shard(rows, num_shards)
    for i, shard_rows in enumerate(shards):
        shard_path = direction_dir / f"shard_{i:03d}.jsonl"
        _write_jsonl(shard_path, shard_rows)


def _maybe_extract_proof_terms(args: argparse.Namespace) -> None:
    """
    Ensure canonical proof terms exist before loading benchmark entries.

    This is only relevant when building directly from `benchmark_root`
    (i.e. no explicit --input provided).
    """
    if args.input:
        return

    policy = str(args.proof_term_extraction).lower()
    if policy not in {"always", "auto", "never"}:
        raise ValueError(f"Invalid proof-term extraction policy: {args.proof_term_extraction}")

    benchmark_root = Path(args.benchmark_root)
    proof_term_dir = benchmark_root / "proof_terms"
    has_indexes = (proof_term_dir / "lean.json").exists() and (proof_term_dir / "rocq.json").exists()

    if policy == "never":
        return
    if policy == "auto" and has_indexes:
        return

    stems_override = None
    if args.proof_term_stems:
        stems_override = [item.strip() for item in str(args.proof_term_stems).split(",") if item.strip()]

    attempts = max(1, int(args.proof_term_attempts))
    retry_sleep_sec = max(0.0, float(args.proof_term_retry_sleep))

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            BenchmarkProofTermExtractionJob(
                benchmark_root=benchmark_root,
                output_dir=proof_term_dir,
                rocq_timeout=int(args.proof_term_rocq_timeout),
                rocq_stem_retries=int(args.proof_term_rocq_stem_retries),
                rocq_stem_retry_sleep=float(args.proof_term_rocq_stem_retry_sleep),
                lean_stem_retries=int(args.proof_term_lean_stem_retries),
                lean_stem_retry_sleep=float(args.proof_term_lean_stem_retry_sleep),
            ).run(
                stems_override=stems_override,
                max_stems=args.proof_term_max_stems,
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            wait_s = retry_sleep_sec * attempt
            print(
                f"[prepare] proof-term extraction failed (attempt {attempt}/{attempts}): {exc}. "
                f"Retrying in {wait_s:.1f}s..."
            )
            if wait_s > 0:
                time.sleep(wait_s)

    assert last_error is not None
    raise last_error


def _make_dataset_card(output_dir: Path, stats: Dict[str, int]) -> None:
    readme = f"""# Babel Benchmark (Prepared)

This folder contains benchmark files prepared for Jean Zay SLURM runs and Hugging Face upload.

## Files
- `lean_to_rocq.jsonl`: Lean -> Rocq direction.
- `rocq_to_lean.jsonl`: Rocq -> Lean direction.
- `lean_to_rocq/shard_XXX.jsonl`: array-job shards.
- `rocq_to_lean/shard_XXX.jsonl`: array-job shards.
- `benchmark_pairs.json`: original paired benchmark entries.

## Sizes
- lean_to_rocq: {stats['lean_to_rocq']} examples
- rocq_to_lean: {stats['rocq_to_lean']} examples
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def _upload_to_hf(output_dir: Path, repo_id: str, private: bool, token_env: str, revision: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "huggingface_hub is required for upload. Install it or run without --hf-repo-id."
        ) from exc

    token = os.getenv(token_env)
    if not token:
        raise RuntimeError(f"Missing token in env var {token_env}.")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=".",
        revision=revision,
        commit_message="Upload benchmark prepared for SLURM + evaluation",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare benchmark files for SLURM and optional HF upload.")
    parser.add_argument("--input", default=None, help="Optional benchmark file. Defaults to new_benchmark parser.")
    parser.add_argument("--benchmark-root", default="new_benchmark")
    parser.add_argument("--output-dir", default="benchmark/prepared")
    parser.add_argument("--num-shards", type=int, default=1, help="Shard count per direction for job arrays.")
    parser.add_argument(
        "--proof-term-extraction",
        choices=("always", "auto", "never"),
        default="always",
        help=(
            "Proof-term extraction policy before preparation (only used when --input is not set). "
            "`always`: run extraction; `auto`: only if proof_terms/*.json missing; `never`: skip."
        ),
    )
    parser.add_argument(
        "--proof-term-rocq-timeout",
        type=int,
        default=120,
        help="Rocq diagnostics timeout (seconds) for extraction.",
    )
    parser.add_argument(
        "--proof-term-stems",
        default=None,
        help="Optional comma-separated subset of source stems for extraction.",
    )
    parser.add_argument(
        "--proof-term-max-stems",
        type=int,
        default=None,
        help="Optional cap on number of stems for extraction.",
    )
    parser.add_argument(
        "--proof-term-lean-stem-retries",
        type=int,
        default=4,
        help="Per-stem retries for Lean extraction on transient empty/missing theorem terms.",
    )
    parser.add_argument(
        "--proof-term-lean-stem-retry-sleep",
        type=float,
        default=2.0,
        help="Base sleep (seconds) between Lean stem retries (linear backoff).",
    )
    parser.add_argument(
        "--proof-term-rocq-stem-retries",
        type=int,
        default=4,
        help="Per-stem retries for Rocq extraction on transient diagnostics failures.",
    )
    parser.add_argument(
        "--proof-term-rocq-stem-retry-sleep",
        type=float,
        default=2.0,
        help="Base sleep (seconds) between Rocq stem retries (linear backoff).",
    )
    parser.add_argument(
        "--proof-term-attempts",
        type=int,
        default=4,
        help="Maximum number of full extraction attempts before failing.",
    )
    parser.add_argument(
        "--proof-term-retry-sleep",
        type=float,
        default=20.0,
        help="Base sleep (seconds) between extraction attempts. Backoff is linear by attempt.",
    )

    parser.add_argument("--hf-repo-id", default=None, help="Optional HF dataset repo id for upload.")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--hf-revision", default="main")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _maybe_extract_proof_terms(args)

    pair_entries = load_benchmark_entries(input_path=args.input, benchmark_root=args.benchmark_root)
    _write_json(output_dir / "benchmark_pairs.json", pair_entries)

    lean_to_rocq_examples = load_examples(args.input, args.benchmark_root, direction="lean_to_rocq")
    rocq_to_lean_examples = load_examples(args.input, args.benchmark_root, direction="rocq_to_lean")

    lean_to_rocq_rows = [_example_to_row(example) for example in lean_to_rocq_examples]
    rocq_to_lean_rows = [_example_to_row(example) for example in rocq_to_lean_examples]

    _write_jsonl(output_dir / "lean_to_rocq.jsonl", lean_to_rocq_rows)
    _write_jsonl(output_dir / "rocq_to_lean.jsonl", rocq_to_lean_rows)

    lean_dir = output_dir / "lean_to_rocq"
    rocq_dir = output_dir / "rocq_to_lean"
    lean_dir.mkdir(parents=True, exist_ok=True)
    rocq_dir.mkdir(parents=True, exist_ok=True)

    _write_shards(lean_dir, lean_to_rocq_rows, args.num_shards)
    _write_shards(rocq_dir, rocq_to_lean_rows, args.num_shards)

    stats = {
        "lean_to_rocq": len(lean_to_rocq_rows),
        "rocq_to_lean": len(rocq_to_lean_rows),
    }
    _make_dataset_card(output_dir, stats)

    if args.hf_repo_id:
        _upload_to_hf(
            output_dir=output_dir,
            repo_id=args.hf_repo_id,
            private=args.hf_private,
            token_env=args.hf_token_env,
            revision=args.hf_revision,
        )

    print(json.dumps({"output_dir": str(output_dir), "stats": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
