from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_BENCHMARK_ROOT = Path("new_benchmark")
DEFAULT_LEAN_WORKSPACE = DEFAULT_BENCHMARK_ROOT / "lean" / "src"
DEFAULT_ROCQ_WORKSPACE = DEFAULT_BENCHMARK_ROOT / "rocq"
DEFAULT_PROOF_TERM_DIR = "proof_terms"

_LEAN_DECL_RE = re.compile(r"^\s*(?:theorem|lemma)\s+([A-Za-z0-9_']+)\b")
_ROCQ_DECL_RE = re.compile(r"^\s*(?:Lemma|Theorem)\s+([A-Za-z0-9_']+)\b")


class ProofTermExtractionError(RuntimeError):
    """Raised when a canonical proof term cannot be loaded for an entry."""


@dataclass(frozen=True)
class LeanDeclaration:
    name: str
    statement: str
    proof: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class RocqDeclaration:
    name: str
    statement: str
    proof: str


def _normalize_term(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(str(chunk) for chunk in value).strip()
    if value is None:
        return ""
    return str(value).strip()


def _load_term_index(root: Path, language: str) -> Dict[Tuple[str, str], Dict[str, object]]:
    path = root / DEFAULT_PROOF_TERM_DIR / f"{language}.json"
    if not path.exists():
        raise ProofTermExtractionError(
            f"Missing proof-term index: {path}. "
            f"Create canonical extracted terms before preparing benchmark files."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    index: Dict[Tuple[str, str], Dict[str, object]] = {}

    def add_term(
        source_stem: str,
        theorem_name: str,
        term: object,
        dependencies: object = None,
    ) -> None:
        source = str(source_stem).strip()
        name = str(theorem_name).strip()
        term_text = _normalize_term(term)
        if not source or not name or not term_text:
            return
        artifact: Dict[str, object] = {"term": term_text}
        if dependencies is not None:
            artifact["dependencies"] = dependencies
        index[(source, name)] = artifact

    if isinstance(payload, dict):
        entries = payload.get("entries")
        if isinstance(entries, list):
            payload = entries
        else:
            # Nested map form: { "<source_stem>": { "<name>": "<term>" } }
            for source_stem, theorem_map in payload.items():
                if not isinstance(theorem_map, dict):
                    continue
                for theorem_name, term in theorem_map.items():
                    if isinstance(term, dict):
                        add_term(
                            source_stem,
                            theorem_name,
                            term.get("term"),
                            term.get("dependencies"),
                        )
                    else:
                        add_term(source_stem, theorem_name, term)
            return index

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            source_stem = row.get("source_stem") or row.get("source")
            theorem_name = row.get("name") or row.get("theorem_name")
            term = row.get("term")
            dependencies = row.get("dependencies")
            add_term(str(source_stem or ""), str(theorem_name or ""), term, dependencies)
        return index

    raise ProofTermExtractionError(
        f"Unsupported proof-term index format in {path}. "
        "Use a list of objects with {source_stem, name, term} or a nested map "
        "{source_stem: {name: term}}."
    )


def _split_lean_block(block: str) -> Tuple[str, str]:
    if ":= by" in block:
        statement, proof_body = block.split(":= by", maxsplit=1)
        return statement.strip(), ("by" + proof_body).strip()
    if ":=" in block:
        statement, proof_body = block.split(":=", maxsplit=1)
        return statement.strip(), proof_body.strip()
    return block.strip(), ""


def _format_dependencies(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        lines: List[str] = []
        for dep in value:
            if isinstance(dep, dict):
                dep_name = str(dep.get("fqn") or dep.get("name") or "").strip()
                dep_type = _normalize_term(dep.get("type"))
                if dep_name and dep_type:
                    lines.append(f"{dep_name} : {dep_type}")
                elif dep_name:
                    lines.append(dep_name)
            else:
                text = _normalize_term(dep)
                if text:
                    lines.append(text)
        return "\n".join(lines).strip()
    return _normalize_term(value)


def _split_rocq_block(block: str) -> Tuple[str, str]:
    if "Proof." in block:
        statement, remainder = block.split("Proof.", maxsplit=1)
        if "Qed." in remainder:
            proof, _ = remainder.rsplit("Qed.", maxsplit=1)
            return statement.strip(), proof.strip()
        return statement.strip(), remainder.strip()
    first_dot = block.find(".")
    if first_dot >= 0:
        return block[: first_dot + 1].strip(), block[first_dot + 1 :].strip()
    return block.strip(), ""


def _collect_blocks(lines: Sequence[str], pattern: re.Pattern[str]) -> List[Tuple[str, int, int]]:
    starts: List[Tuple[str, int]] = []
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            starts.append((match.group(1), idx))

    blocks: List[Tuple[str, int, int]] = []
    for i, (name, start) in enumerate(starts):
        end = starts[i + 1][1] - 1 if i + 1 < len(starts) else len(lines) - 1
        blocks.append((name, start, end))
    return blocks


def parse_lean_declarations(path: Path) -> List[LeanDeclaration]:
    lines = path.read_text(encoding="utf-8").splitlines()
    decls: List[LeanDeclaration] = []
    for name, start, end in _collect_blocks(lines, _LEAN_DECL_RE):
        block = "\n".join(lines[start : end + 1]).strip()
        statement, proof = _split_lean_block(block)
        decls.append(
            LeanDeclaration(
                name=name,
                statement=statement,
                proof=proof,
                line_start=start,
                line_end=end,
            )
        )
    return decls


def parse_rocq_declarations(path: Path) -> List[RocqDeclaration]:
    lines = path.read_text(encoding="utf-8").splitlines()
    decls: List[RocqDeclaration] = []
    for name, start, end in _collect_blocks(lines, _ROCQ_DECL_RE):
        block = "\n".join(lines[start : end + 1]).strip()
        statement, proof = _split_rocq_block(block)
        decls.append(RocqDeclaration(name=name, statement=statement, proof=proof))
    return decls


def get_workspace(kind: str, benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT) -> Path:
    root = Path(benchmark_root)
    kind = kind.lower()
    if kind == "lean":
        return root / "lean" / "src"
    if kind == "rocq":
        return root / "rocq"
    raise ValueError(f"Unknown workspace kind: {kind}")


def build_new_benchmark_entries(benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT) -> List[dict]:
    root = Path(benchmark_root)
    lean_dir = get_workspace("lean", root)
    rocq_dir = get_workspace("rocq", root)
    lean_terms = _load_term_index(root, "lean")
    rocq_terms = _load_term_index(root, "rocq")

    lean_files = {path.stem: path for path in lean_dir.glob("*.lean")}
    rocq_files = {path.stem: path for path in rocq_dir.glob("*.v")}
    common_stems = sorted(set(lean_files).intersection(rocq_files))

    entries: List[dict] = []
    for stem in common_stems:
        lean_decls = parse_lean_declarations(lean_files[stem])
        rocq_decls = parse_rocq_declarations(rocq_files[stem])
        rocq_by_name: Dict[str, RocqDeclaration] = {decl.name: decl for decl in rocq_decls}

        for lean_decl in lean_decls:
            rocq_decl = rocq_by_name.get(lean_decl.name)
            if rocq_decl is None:
                continue

            lean_key = (stem, lean_decl.name)
            rocq_key = (stem, rocq_decl.name)

            lean_artifact = lean_terms.get(lean_key)
            lean_term = _normalize_term((lean_artifact or {}).get("term"))
            if not lean_term:
                raise ProofTermExtractionError(
                    f"Missing Lean proof term for source_stem={stem}, theorem={lean_decl.name}."
                )

            rocq_artifact = rocq_terms.get(rocq_key)
            rocq_term = _normalize_term((rocq_artifact or {}).get("term"))
            if not rocq_term:
                raise ProofTermExtractionError(
                    f"Missing Rocq proof term for source_stem={stem}, theorem={rocq_decl.name}."
                )

            entries.append(
                {
                    "name": lean_decl.name,
                    "source_stem": stem,
                    "lean": {
                        "name": lean_decl.name,
                        "source": stem,
                        "statement": lean_decl.statement,
                        "proof": lean_decl.proof,
                        "term": lean_term,
                        "dependencies": _format_dependencies(
                            (lean_artifact or {}).get("dependencies")
                        ),
                        "initial_goal": [lean_decl.statement],
                        "lines": [lean_decl.line_start, lean_decl.line_end],
                    },
                    "rocq": {
                        "name": rocq_decl.name,
                        "source": stem,
                        "statement": rocq_decl.statement,
                        "proof": rocq_decl.proof,
                        "term": rocq_term,
                        "dependencies": _format_dependencies(
                            (rocq_artifact or {}).get("dependencies")
                        ),
                        "initial_goal": [rocq_decl.statement],
                    },
                }
            )
    return entries


def load_benchmark_entries(
    input_path: Optional[str] = None,
    benchmark_root: Path | str = DEFAULT_BENCHMARK_ROOT,
) -> List[dict]:
    if input_path:
        with open(input_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return build_new_benchmark_entries(benchmark_root=benchmark_root)
