from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.common.new_benchmark import get_workspace, load_benchmark_entries


@dataclass(frozen=True)
class DirectionConfig:
    name: str
    source_field: str
    target_field: str
    source_lang: str
    target_lang: str


DIRECTION_CONFIGS: Dict[str, DirectionConfig] = {
    "lean_to_rocq": DirectionConfig(
        name="lean_to_rocq",
        source_field="lean",
        target_field="rocq",
        source_lang="lean",
        target_lang="rocq",
    ),
    "rocq_to_lean": DirectionConfig(
        name="rocq_to_lean",
        source_field="rocq",
        target_field="lean",
        source_lang="rocq",
        target_lang="lean",
    ),
}


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    direction: str
    theorem_name: str
    source_stem: str
    source_term: str
    dependencies: str
    target_statement: str
    target_initial_goals: List[str]
    target_lines: Optional[Tuple[int, int]]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_BOX_RE = re.compile(r"\\box\{(.*?)\}", re.DOTALL)


def _as_text(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    if value is None:
        return ""
    return str(value)


def _as_goals(value: object, fallback_statement: str) -> List[str]:
    if isinstance(value, list):
        goals = [str(x).strip() for x in value if str(x).strip()]
        if goals:
            return goals
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if fallback_statement.strip():
        return [fallback_statement.strip()]
    return [""]


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_raw_entries(input_path: Optional[str], benchmark_root: str) -> List[dict]:
    if input_path is None:
        return load_benchmark_entries(input_path=None, benchmark_root=benchmark_root)

    path = Path(input_path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return _read_jsonl(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported benchmark payload in {path}. Expected a list.")


def _from_pair_entry(entry: dict, direction: str) -> Optional[BenchmarkExample]:
    config = DIRECTION_CONFIGS[direction]
    source = entry.get(config.source_field)
    target = entry.get(config.target_field)
    if not isinstance(source, dict) or not isinstance(target, dict):
        return None

    theorem_name = str(target.get("name") or source.get("name") or entry.get("name") or "").strip()
    if not theorem_name:
        return None

    source_stem = str(target.get("source") or source.get("source") or entry.get("source_stem") or "").strip()
    if not source_stem:
        return None

    source_term = _as_text(source.get("term")).strip()
    if not source_term:
        raise ValueError(
            f"Missing source proof term for theorem={theorem_name}, direction={direction}. "
            "This pipeline requires canonical extracted proof terms."
        )
    dependencies = _as_text(target.get("dependencies")).strip()
    target_statement = _as_text(target.get("statement")).strip()
    target_initial_goals = _as_goals(target.get("initial_goal"), target_statement)

    target_lines: Optional[Tuple[int, int]] = None
    if config.target_field == "lean":
        raw_lines = target.get("lines")
        if isinstance(raw_lines, list) and len(raw_lines) == 2:
            try:
                target_lines = (int(raw_lines[0]), int(raw_lines[1]))
            except (TypeError, ValueError):
                target_lines = None

    example_id = f"{source_stem}:{theorem_name}:{direction}"
    return BenchmarkExample(
        example_id=example_id,
        direction=direction,
        theorem_name=theorem_name,
        source_stem=source_stem,
        source_term=source_term,
        dependencies=dependencies,
        target_statement=target_statement,
        target_initial_goals=target_initial_goals,
        target_lines=target_lines,
    )


def _from_flat_row(row: dict, direction: str) -> Optional[BenchmarkExample]:
    row_direction = str(row.get("direction") or "").strip()
    if row_direction and row_direction != direction:
        return None

    theorem_name = str(row.get("theorem_name") or row.get("name") or "").strip()
    source_stem = str(row.get("source_stem") or row.get("source") or "").strip()
    source_term = _as_text(row.get("source_term") or row.get("term")).strip()
    dependencies = _as_text(row.get("dependencies")).strip()
    target_statement = _as_text(row.get("target_statement") or row.get("statement")).strip()

    if not theorem_name or not source_stem:
        return None
    if not source_term:
        raise ValueError(
            f"Missing source proof term for theorem={theorem_name}, direction={direction}. "
            "This pipeline requires canonical extracted proof terms."
        )

    goals = _as_goals(row.get("target_initial_goals") or row.get("initial_goal"), target_statement)

    target_lines: Optional[Tuple[int, int]] = None
    raw_lines = row.get("target_lines") or row.get("lines")
    if isinstance(raw_lines, list) and len(raw_lines) == 2:
        try:
            target_lines = (int(raw_lines[0]), int(raw_lines[1]))
        except (TypeError, ValueError):
            target_lines = None

    example_id = f"{source_stem}:{theorem_name}:{direction}"
    return BenchmarkExample(
        example_id=example_id,
        direction=direction,
        theorem_name=theorem_name,
        source_stem=source_stem,
        source_term=source_term,
        dependencies=dependencies,
        target_statement=target_statement,
        target_initial_goals=goals,
        target_lines=target_lines,
    )


def load_examples(input_path: Optional[str], benchmark_root: str, direction: str) -> List[BenchmarkExample]:
    rows = _load_raw_entries(input_path=input_path, benchmark_root=benchmark_root)
    examples: List[BenchmarkExample] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        # New benchmark pair entry
        if "lean" in row and "rocq" in row:
            item = _from_pair_entry(row, direction)
            if item is not None:
                examples.append(item)
            continue

        # Flattened row (e.g., JSONL export)
        item = _from_flat_row(row, direction)
        if item is not None:
            examples.append(item)

    return examples


def resolve_workspace(kind: str, benchmark_root: str, override: Optional[str]) -> str:
    if override:
        return str(Path(override).resolve())
    return str(get_workspace(kind, benchmark_root=Path(benchmark_root)).resolve())


def load_instruction(prompt_path: str) -> str:
    payload = json.loads(Path(prompt_path).read_text(encoding="utf-8"))
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Prompt file {prompt_path} does not contain a valid `instruction` string.")
    return instruction


def build_instruction(prompt_instruction: str, disable_reasoning: bool) -> str:
    if not disable_reasoning:
        return prompt_instruction

    # Keep the structure simple and deterministic for no-reasoning ablations.
    return (
        "You are given a proof term:\n\n{term}\n\n"
        "Your task is to derive a sequence of tactics that corresponds to this term.\n\n"
        "Do not include any reasoning text. Return only tactic script blocks in the form:\n\n"
        "\\box{{\n"
        "  <tactic>\n"
        "}}\n\n"
        "Some dependencies that could be helpful:\n\n"
        "{dependencies}"
    )


def build_initial_think(direction: str, goals: Iterable[str]) -> str:
    goal_text = "\n".join(str(g) for g in goals)
    if direction == "rocq_to_lean":
        return (
            "<think> Okay, let's try to transform this proof term into a sequence "
            "of Lean 4 tactics. First let's write down the hypotheses, and the "
            "initial goal given by the Lean proof assistant:\n"
            f"{goal_text}."
        )
    return (
        "<think> Okay, let's try to transform this proof term into a sequence "
        "of Rocq tactics. First let's write down the hypotheses, and the "
        "initial goal (after the \"|-\" symbol) given by the Rocq proof assistant:\n"
        f"{goal_text}."
    )


def extract_proof_blocks(model_text: str) -> List[str]:
    blocks = [chunk.strip() for chunk in _BOX_RE.findall(model_text) if chunk.strip()]
    if blocks:
        return blocks

    no_think = _THINK_RE.sub(" ", model_text)
    fallback = no_think.strip()
    return [fallback] if fallback else []


def blocks_to_proof(blocks: List[str], target_lang: str) -> str:
    if not blocks:
        return ""
    if target_lang == "lean":
        return "\n".join(blocks).strip()

    # Rocq/Coq sentence style: keep author's dots if present, but normalize by line.
    merged = "\n".join(blocks).strip()
    return merged


def short_text(text: str, limit: int = 2000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"
