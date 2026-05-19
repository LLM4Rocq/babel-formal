from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.evaluator.prover import MessageType

from .utils import (
    DIRECTION_CONFIGS,
    BenchmarkExample,
    blocks_to_proof,
    build_initial_think,
    build_instruction,
    extract_proof_blocks,
    load_examples,
    load_instruction,
    resolve_workspace,
    short_text,
)
from .verifiers import BaseVerifier, build_lean_verifier, build_rocq_verifier


_STEP_RE = re.compile(r"(?:step|global_step|iter|epoch)[^0-9]*([0-9]+)", re.IGNORECASE)


def _safe_model_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value).strip("_")


def _looks_like_hf_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = bool(list(path.glob("*.safetensors"))) or bool(list(path.glob("pytorch_model*.bin")))
    return has_config and has_weights


def _candidate_score(path: Path) -> tuple[int, float]:
    name = path.as_posix()
    step_match = _STEP_RE.search(name)
    step = int(step_match.group(1)) if step_match else -1
    return (step, path.stat().st_mtime)


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if not path.exists():
        return model_path

    if _looks_like_hf_checkpoint(path):
        return str(path)

    candidates: List[Path] = []
    for root, _dirs, _files in os.walk(path):
        current = Path(root)
        # Keep search bounded for speed/readability.
        rel_depth = len(current.relative_to(path).parts)
        if rel_depth > 4:
            continue
        if _looks_like_hf_checkpoint(current):
            candidates.append(current)

    if not candidates:
        return str(path)

    candidates.sort(key=_candidate_score, reverse=True)
    return str(candidates[0])


def resolve_tokenizer_path(tokenizer_path: Optional[str], model_path: str) -> str:
    if tokenizer_path:
        return tokenizer_path

    path = Path(model_path)
    if (path / "tokenizer.json").exists() or (path / "tokenizer_config.json").exists():
        return str(path)

    # Common NeMo AutoModel checkpoint layout candidate.
    consolidated = path / "model" / "consolidated"
    if consolidated.exists() and ((consolidated / "tokenizer.json").exists() or (consolidated / "tokenizer_config.json").exists()):
        return str(consolidated)

    return str(path)


class VLLMGenerator:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
    ):
        try:
            from transformers import AutoTokenizer
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Missing dependencies for generation. Install `transformers` and `vllm` in the benchmark env."
            ) from exc

        self._SamplingParams = __import__("vllm", fromlist=["SamplingParams"]).SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            max_num_seqs=16,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

    def generate_one(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Optional[List[str]],
        forced_prefix: str = "",
    ) -> str:
        sampling_params = self._SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
            stop=stop,
        )

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if forced_prefix:
            prompt_text += forced_prefix

        outputs = self.llm.generate([prompt_text], sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        return str(outputs[0].outputs[0].text)


def _make_feedback_message(
    feedback_level: str,
    proof: str,
    goals: List[str],
    error: str,
) -> str:
    lines: List[str] = ["Previous attempt did not solve the theorem."]

    if feedback_level in {"errors", "goals_errors"} and error.strip():
        lines.append("Error:")
        lines.append(short_text(error, limit=1200))

    if feedback_level in {"goals", "goals_errors"} and goals:
        lines.append("Current goals:")
        lines.append(short_text("\n\n".join(goals), limit=1200))

    if feedback_level == "no_feedback":
        lines.append("Try a different tactic plan.")

    lines.append("Attempted proof:")
    lines.append(short_text(proof, limit=1200))
    lines.append("Provide a corrected proof.")
    return "\n\n".join(lines)


def _rollback_proof(proof: str) -> str:
    blocks = extract_proof_blocks(proof)
    if len(blocks) <= 1:
        return ""
    return blocks_to_proof(blocks[:-1], target_lang="lean")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_verifier(
    direction: str,
    rocq_backend: str,
    lean_workspace: str,
    rocq_workspace: str,
    kimina_url: str,
    kimina_timeout: int,
    kimina_api_key: Optional[str],
    rocq_host: str,
    rocq_port: int,
    rocq_timeout_per_step: int,
    rocq_start_server: bool,
) -> BaseVerifier:
    if direction == "rocq_to_lean":
        return build_lean_verifier(
            workspace=lean_workspace,
            kimina_url=kimina_url,
            kimina_timeout=kimina_timeout,
            kimina_api_key=kimina_api_key,
        )

    return build_rocq_verifier(
        backend=rocq_backend,
        workspace=rocq_workspace,
        host=rocq_host,
        port=rocq_port,
        timeout_per_step=rocq_timeout_per_step,
        start_server=rocq_start_server,
    )


def _target_lang(direction: str) -> str:
    return DIRECTION_CONFIGS[direction].target_lang


def run_direction(
    direction: str,
    examples: List[BenchmarkExample],
    instruction_template: str,
    generator: VLLMGenerator,
    verifier: BaseVerifier,
    output_dir: Path,
    pass_k: int,
    max_rounds: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    feedback_level: str,
    disable_reasoning: bool,
    seed: int,
) -> dict:
    random.Random(seed).shuffle(examples)

    attempts_path = output_dir / f"attempts_{direction}.jsonl"
    results_path = output_dir / f"results_{direction}.jsonl"

    success_count = 0
    per_example_results: List[dict] = []

    for example in tqdm(examples, desc=f"{direction} examples"):
        example_success = False
        winning_candidate: Optional[int] = None
        candidate_summaries: List[dict] = []

        for candidate_idx in range(pass_k):
            messages: List[Dict[str, str]] = [
                {
                    "role": "user",
                    "content": instruction_template.format(
                        term=example.source_term,
                        dependencies=example.dependencies,
                    ),
                }
            ]

            current_prefix = ""
            if not disable_reasoning:
                current_prefix = build_initial_think(direction, example.target_initial_goals)

            candidate_success = False
            last_error = ""

            for round_idx in range(max_rounds):
                started = time.time()
                output_text = generator.generate_one(
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=["<think>"] if not disable_reasoning else None,
                    forced_prefix=current_prefix,
                )
                current_prefix = ""

                blocks = extract_proof_blocks(output_text)
                proof_text = blocks_to_proof(blocks, target_lang=_target_lang(direction))

                verification = verifier.verify(example, proof_text)
                elapsed = time.time() - started

                status_name = verification.status.name
                attempt_payload = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "direction": direction,
                    "example_id": example.example_id,
                    "theorem_name": example.theorem_name,
                    "source_stem": example.source_stem,
                    "candidate_idx": candidate_idx,
                    "round_idx": round_idx,
                    "status": status_name,
                    "elapsed_seconds": round(elapsed, 4),
                    "output": output_text,
                    "proof": proof_text,
                    "goals": verification.goals,
                    "error": verification.error,
                    "messages": messages,
                    "raw_verifier": verification.raw,
                }
                _append_jsonl(attempts_path, attempt_payload)

                if verification.status == MessageType.FINISH:
                    candidate_success = True
                    example_success = True
                    winning_candidate = candidate_idx
                    break

                last_error = verification.error

                if feedback_level == "whole_proof":
                    break

                feedback_message = _make_feedback_message(
                    feedback_level=feedback_level,
                    proof=proof_text,
                    goals=verification.goals,
                    error=verification.error,
                )

                if feedback_level in {"errors", "goals_errors"}:
                    rolled_back = _rollback_proof(proof_text)
                    if rolled_back:
                        feedback_message += "\n\nRollback prefix kept:\n" + short_text(rolled_back, limit=1200)

                messages.append({"role": "assistant", "content": output_text})
                messages.append({"role": "user", "content": feedback_message})

            candidate_summaries.append(
                {
                    "candidate_idx": candidate_idx,
                    "success": candidate_success,
                    "last_error": last_error,
                }
            )

            if candidate_success:
                break

        if example_success:
            success_count += 1

        result_payload = {
            "direction": direction,
            "example_id": example.example_id,
            "theorem_name": example.theorem_name,
            "source_stem": example.source_stem,
            "success": example_success,
            "winning_candidate": winning_candidate,
            "candidate_summaries": candidate_summaries,
        }
        _append_jsonl(results_path, result_payload)
        per_example_results.append(result_payload)

    total = len(examples)
    pass_at_k = float(success_count) / float(total) if total else 0.0

    return {
        "direction": direction,
        "total_examples": total,
        "successes": success_count,
        "pass_at_k": pass_at_k,
        "k": pass_k,
        "max_rounds": max_rounds,
        "feedback_level": feedback_level,
        "disable_reasoning": disable_reasoning,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lean/Rocq translation benchmark with feedback loops.")
    parser.add_argument("--direction", choices=["lean_to_rocq", "rocq_to_lean", "both"], default="both")
    parser.add_argument("--input", default=None, help="Optional benchmark file (.json or .jsonl).")
    parser.add_argument("--benchmark-root", default="new_benchmark", help="Benchmark root when --input is omitted.")

    parser.add_argument("--model-path", required=True, help="Model path or HF id. Supports NeMo checkpoint dirs.")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path. Defaults to resolved model path.")

    parser.add_argument("--prompt-lean-path", default="config/prompts/prompt_lean.json")
    parser.add_argument("--prompt-rocq-path", default="config/prompts/prompt_rocq.json")
    parser.add_argument("--disable-reasoning", action="store_true", help="Use no-reasoning prompt variant.")

    parser.add_argument("--pass-k", type=int, default=32)
    parser.add_argument("--max-rounds", type=int, default=2, help="Feedback retries per candidate.")
    parser.add_argument(
        "--feedback-level",
        choices=["whole_proof", "goals", "errors", "goals_errors", "no_feedback"],
        default="goals",
        help="Feedback policy from paper/icml.",
    )

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)

    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=12000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)

    parser.add_argument("--output-dir", default="export/benchmark_runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=1111)

    parser.add_argument("--lean-workspace", default=None)
    parser.add_argument("--rocq-workspace", default=None)

    parser.add_argument("--lean-kimina-url", default="http://127.0.0.1:8000")
    parser.add_argument("--lean-kimina-timeout", type=int, default=300)
    parser.add_argument("--lean-kimina-api-key-env", default="LEAN_SERVER_API_KEY")

    parser.add_argument("--rocq-backend", choices=["pytanque", "ml_server"], default="pytanque")
    parser.add_argument("--rocq-host", default="127.0.0.1")
    parser.add_argument("--rocq-port", type=int, default=8765)
    parser.add_argument("--rocq-timeout-per-step", type=int, default=10)
    parser.add_argument("--rocq-start-server", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    random.seed(args.seed)

    resolved_model_path = resolve_model_path(args.model_path)
    resolved_tokenizer_path = resolve_tokenizer_path(args.tokenizer_path, resolved_model_path)

    lean_workspace = resolve_workspace("lean", args.benchmark_root, args.lean_workspace)
    rocq_workspace = resolve_workspace("rocq", args.benchmark_root, args.rocq_workspace)

    model_tag = _safe_model_tag(Path(resolved_model_path).name or resolved_model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{timestamp}_{model_tag}"
    output_dir = Path(args.output_dir) / run_name
    _ensure_dir(output_dir)

    config_dump = {
        "args": vars(args),
        "resolved_model_path": resolved_model_path,
        "resolved_tokenizer_path": resolved_tokenizer_path,
        "lean_workspace": lean_workspace,
        "rocq_workspace": rocq_workspace,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config_dump, indent=2), encoding="utf-8")

    generator = VLLMGenerator(
        model_path=resolved_model_path,
        tokenizer_path=resolved_tokenizer_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    directions = [args.direction] if args.direction != "both" else ["lean_to_rocq", "rocq_to_lean"]

    summaries: List[dict] = []
    for direction in directions:
        prompt_path = args.prompt_rocq_path if direction == "lean_to_rocq" else args.prompt_lean_path
        instruction = load_instruction(prompt_path)
        instruction = build_instruction(instruction, disable_reasoning=args.disable_reasoning)

        examples = load_examples(args.input, args.benchmark_root, direction=direction)

        kimina_api_key = os.getenv(args.lean_kimina_api_key_env)
        verifier = _build_verifier(
            direction=direction,
            rocq_backend=args.rocq_backend,
            lean_workspace=lean_workspace,
            rocq_workspace=rocq_workspace,
            kimina_url=args.lean_kimina_url,
            kimina_timeout=args.lean_kimina_timeout,
            kimina_api_key=kimina_api_key,
            rocq_host=args.rocq_host,
            rocq_port=args.rocq_port,
            rocq_timeout_per_step=args.rocq_timeout_per_step,
            rocq_start_server=args.rocq_start_server,
        )

        try:
            summary = run_direction(
                direction=direction,
                examples=examples,
                instruction_template=instruction,
                generator=generator,
                verifier=verifier,
                output_dir=output_dir,
                pass_k=args.pass_k,
                max_rounds=args.max_rounds,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                feedback_level=args.feedback_level,
                disable_reasoning=args.disable_reasoning,
                seed=args.seed,
            )
            summaries.append(summary)
        finally:
            verifier.close()

    final_summary = {
        "run_name": run_name,
        "resolved_model_path": resolved_model_path,
        "resolved_tokenizer_path": resolved_tokenizer_path,
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    print(json.dumps(final_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
