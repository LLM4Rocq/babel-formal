from __future__ import annotations

import argparse
import inspect
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_COQPYT = _REPO_ROOT / "coqpyt"
if _VENDORED_COQPYT.exists():
    sys.path.insert(0, str(_VENDORED_COQPYT))

from coqpyt.coq.proof_file import _AuxFile

from src.common.new_benchmark import (
    get_workspace,
    parse_lean_declarations,
    parse_rocq_declarations,
)
from src.lean_rocq_translation.step_4.exec import LSPNavigator, TheoremExtractor


_ROCQ_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*$")
_ROCQ_SKIP_LINES = (
    "Closed under the global context",
    "Fetching opaque proofs from disk",
)
_ROCQ_RESERVED_IDENTIFIERS = {
    "forall",
    "fun",
    "fix",
    "let",
    "in",
    "match",
    "with",
    "end",
    "if",
    "then",
    "else",
    "Type",
    "Prop",
    "Set",
}
_LEAN_RESERVED_IDENTIFIERS = {
    "theorem",
    "lemma",
    "by",
    "let",
    "have",
    "fun",
    "if",
    "then",
    "else",
    "match",
    "with",
    "_",
}
_LEAN_TERM_IDENT_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*"
)
_ROCQ_TERM_IDENT_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*"
)

logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    pass


def _install_leanclient_compat_patch() -> None:
    """
    Make step_4 extraction compatible with older leanclient builds.

    Some environments expose `SingleFileClient.update_file(changes)` (no `timeout` kwarg),
    while step_4 calls `update_file(..., timeout=...)`.
    """
    try:
        from leanclient import client as lean_client_module
    except Exception:
        return

    single_cls = getattr(lean_client_module, "SingleFileClient", None)
    if single_cls is None:
        return

    update_fn = getattr(single_cls, "update_file", None)
    if update_fn is None:
        return

    try:
        sig = inspect.signature(update_fn)
    except (TypeError, ValueError):
        return

    if "timeout" in sig.parameters:
        return

    original_update_file = update_fn

    def update_file_compat(self, changes, timeout=None):  # type: ignore[no-redef]
        _ = timeout
        return original_update_file(self, changes)

    single_cls.update_file = update_file_compat


@dataclass(frozen=True)
class DependencyInfo:
    name: str
    fqn: str
    type: str


@dataclass(frozen=True)
class TheoremArtifact:
    source_stem: str
    name: str
    term: str
    dependencies: List[DependencyInfo]

    def to_json(self) -> dict:
        return {
            "source_stem": self.source_stem,
            "name": self.name,
            "term": self.term,
            "dependencies": [
                {"name": dep.name, "fqn": dep.fqn, "type": dep.type}
                for dep in self.dependencies
            ],
        }


class RocqLspQuerySession:
    """Small query wrapper around coq-lsp diagnostics, same mechanism as step_0 pipeline."""

    def __init__(self, file_path: Path, workspace: Path, timeout: int = 120):
        self.file_path = file_path
        self.timeout = max(1, int(timeout))
        self._aux = _AuxFile(
            str(file_path),
            copy=True,
            workspace=str(workspace),
            timeout=self.timeout,
        )
        self._aux.didOpen()

    def close(self) -> None:
        self._aux.close()

    def _find_query_message(self, keyword: str, query: str, line: int) -> str | None:
        message = self._aux.get_diagnostics(keyword, query, line)
        if message is not None:
            return str(message)

        # Some coq-lsp diagnostics can land on an adjacent line; probe nearby lines first.
        for delta in (1, -1, 2, -2, 3, -3):
            message = self._aux.get_diagnostics(keyword, query, line + delta)
            if message is not None:
                return str(message)

        # Last-resort fallback: accept same query regardless of exact line.
        # _AuxFile has no public API for this, so we reuse its internal query view.
        try:
            queries = self._aux._AuxFile__get_queries(keyword)  # type: ignore[attr-defined]
        except Exception:
            return None

        for query_item in queries:
            if getattr(query_item, "query", None) != f"{query}":
                continue
            results = getattr(query_item, "results", None) or []
            if not results:
                return None
            return str(getattr(results[-1], "message", ""))
        return None

    def query(
        self,
        keyword: str,
        query: str,
        command: str,
        timeout_sec: float | None = None,
    ) -> str:
        line = len(self._aux.read().split("\n"))
        self._aux.append(f"\n{command}")
        self._aux.didChange()

        deadline = time.monotonic() + (
            float(timeout_sec) if timeout_sec is not None else float(self.timeout)
        )
        sleep_s = 0.05
        while time.monotonic() < deadline:
            message = self._find_query_message(keyword, query, line)
            if message is not None:
                return message
            time.sleep(sleep_s)
            sleep_s = min(0.5, sleep_s * 1.5)

        raise ExtractionError(
            f"Rocq query did not return diagnostics for `{command}` in {self.file_path}"
        )


class LeanTrainingPipelineExtractor:
    """Lean extraction via the same step_4 TheoremExtractor used for training-set prep."""

    def __init__(
        self,
        benchmark_root: Path,
        stem_retries: int = 4,
        retry_sleep_sec: float = 2.0,
    ):
        self.benchmark_root = benchmark_root
        self.workspace = (benchmark_root / "lean").resolve()
        self.lean_src = self.workspace / "src"
        self.stem_retries = max(1, int(stem_retries))
        self.retry_sleep_sec = max(0.0, float(retry_sleep_sec))

    def _decl_names_by_stem(self, stems: Sequence[str]) -> Dict[str, Set[str]]:
        names: Dict[str, Set[str]] = {}
        for stem in stems:
            decls = parse_lean_declarations(self.lean_src / f"{stem}.lean")
            names[stem] = {decl.name for decl in decls}
        return names

    @staticmethod
    def _normalize_term(term: str) -> str:
        text = (term or "").strip()
        if not text:
            return ""
        if ":=" in text:
            return text.split(":=", maxsplit=1)[1].strip()
        return text

    def _build_artifacts_for_stem(
        self,
        stem: str,
        theorem_names: Set[str],
        by_name: Dict[str, object],
        rel_path: str,
        extractor: TheoremExtractor,
        nav_cache: Dict[str, LSPNavigator],
    ) -> List[TheoremArtifact]:
        missing = sorted(theorem_names.difference(by_name.keys()))
        if missing:
            raise ExtractionError(
                f"Lean extraction missing declarations in {stem}: {', '.join(missing)}"
            )

        empty_terms: List[str] = []
        artifacts: List[TheoremArtifact] = []
        for theorem_name in sorted(theorem_names):
            info = by_name[theorem_name]
            term = self._normalize_term(getattr(info, "term", "") or "")
            if not term:
                nav = nav_cache.get(rel_path)
                if nav is None:
                    nav = LSPNavigator(extractor.client, rel_path)
                    nav_cache[rel_path] = nav
                print_text = nav.print_decl_via_diagnostics(theorem_name) or ""
                term = self._normalize_term(print_text)
            if not term:
                empty_terms.append(theorem_name)
                continue

            # Restrict dependency candidates to symbols that actually appear in the
            # canonical extracted term, then keep only likely global references.
            rhs = term.split(":=", maxsplit=1)[1] if ":=" in term else term
            term_candidates: Set[str] = set(_LEAN_TERM_IDENT_RE.findall(rhs))
            dependencies: List[DependencyInfo] = []
            seen: Set[str] = set()
            for symbol in getattr(info, "symbols", []):
                dep_fqn = (symbol.name or "").strip()
                dep_type = (symbol.check_text or symbol.type or "").strip()
                if (
                    not dep_fqn
                    or dep_fqn == theorem_name
                    or dep_fqn in seen
                    or dep_fqn in _LEAN_RESERVED_IDENTIFIERS
                ):
                    continue
                if not dep_type:
                    continue
                # Must be present in term text (directly or by short name).
                dep_short = dep_fqn.split(".")[-1]
                if dep_fqn not in term_candidates and dep_short not in term_candidates:
                    continue
                # Drop obviously local binders/variables.
                if (
                    "." not in dep_fqn
                    and "_" not in dep_fqn
                    and dep_fqn[:1].islower()
                    and len(dep_fqn) <= 3
                ):
                    continue
                seen.add(dep_fqn)
                dependencies.append(
                    DependencyInfo(
                        name=dep_short,
                        fqn=dep_fqn,
                        type=dep_type,
                    )
                )

            artifacts.append(
                TheoremArtifact(
                    source_stem=stem,
                    name=theorem_name,
                    term=term,
                    dependencies=dependencies,
                )
            )

        if empty_terms:
            raise ExtractionError(
                f"Lean theorem has empty proof term in {stem}: {', '.join(empty_terms)}"
            )

        return artifacts

    def _extract_stem_with_retries(
        self,
        extractor: TheoremExtractor,
        rel_path: str,
        stem: str,
        theorem_names: Set[str],
    ) -> List[TheoremArtifact]:
        last_error: Exception | None = None
        nav_cache: Dict[str, LSPNavigator] = {}
        for attempt in range(1, self.stem_retries + 1):
            try:
                infos = extractor.extract_file(rel_path, theorem_names)
                by_name = {info.name: info for info in infos}
                return self._build_artifacts_for_stem(
                    stem=stem,
                    theorem_names=theorem_names,
                    by_name=by_name,
                    rel_path=rel_path,
                    extractor=extractor,
                    nav_cache=nav_cache,
                )
            except ExtractionError as exc:
                last_error = exc
                if attempt >= self.stem_retries:
                    break
                sleep_s = self.retry_sleep_sec * attempt
                logger.warning(
                    "Lean extraction transient failure for stem=%s (attempt %d/%d): %s. Retrying in %.1fs",
                    stem,
                    attempt,
                    self.stem_retries,
                    exc,
                    sleep_s,
                )
                print(
                    f"[extract][lean] transient failure stem={stem} attempt={attempt}/{self.stem_retries}: {exc}. "
                    f"retry_in={sleep_s:.1f}s",
                    flush=True,
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)

        raise ExtractionError(
            f"Lean extraction failed for {stem} after {self.stem_retries} attempts: {last_error}"
        )

    def extract(
        self,
        stems: Sequence[str],
        on_stem_done: Callable[[str, List[TheoremArtifact]], None] | None = None,
    ) -> List[TheoremArtifact]:
        _install_leanclient_compat_patch()
        names_by_stem = self._decl_names_by_stem(stems)
        artifacts: List[TheoremArtifact] = []
        with TemporaryDirectory(prefix="lean_benchmark_extract_") as tmp_root:
            workspace, rel_dir = self._build_temp_workspace(Path(tmp_root))
            extractor = TheoremExtractor(str(workspace))
            try:
                for idx, stem in enumerate(stems, start=1):
                    print(f"[extract][lean] {idx}/{len(stems)} stem={stem}", flush=True)
                    rel_path = f"{rel_dir}/{stem}.lean"
                    stem_artifacts = self._extract_stem_with_retries(
                        extractor=extractor,
                        rel_path=rel_path,
                        stem=stem,
                        theorem_names=names_by_stem[stem],
                    )
                    artifacts.extend(stem_artifacts)
                    if on_stem_done is not None:
                        on_stem_done(stem, stem_artifacts)
            finally:
                extractor.close()

        return artifacts

    def _build_temp_workspace(self, temp_workspace: Path) -> Tuple[Path, str]:
        """
        Create an isolated Lean workspace expected by step_4 extractor.

        The benchmark lean root is not guaranteed to be a valid Lake project as-is
        (notably missing `Dataset/Dataset.lean`). We generate a minimal temporary
        project that points to the benchmark sources without mutating the repo.
        """
        dataset_dir = temp_workspace / "Dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for lean_file in sorted(self.lean_src.glob("*.lean")):
            target = dataset_dir / lean_file.name
            target.write_text(lean_file.read_text(encoding="utf-8"), encoding="utf-8")

        root_module = dataset_dir / "Dataset.lean"
        if not root_module.exists():
            root_module.write_text(
                "-- Auto-generated for benchmark extraction.\nnamespace Dataset\nend Dataset\n",
                encoding="utf-8",
            )

        lakefile_src = self.workspace / "lakefile.toml"
        if lakefile_src.exists():
            (temp_workspace / "lakefile.toml").write_text(
                lakefile_src.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            (temp_workspace / "lakefile.toml").write_text(
                'name = "dataset"\n'
                'defaultTargets = ["Dataset"]\n\n'
                "[[lean_lib]]\n"
                'name = "Dataset"\n'
                'srcDir = "Dataset"\n'
                'roots = ["Dataset"]\n',
                encoding="utf-8",
            )

        lake_manifest_src = self.workspace / "lake-manifest.json"
        if lake_manifest_src.exists():
            (temp_workspace / "lake-manifest.json").write_text(
                lake_manifest_src.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        toolchain_candidates = [
            self.workspace / "lean-toolchain",
            self.lean_src / "lean-toolchain",
        ]
        for candidate in toolchain_candidates:
            if candidate.exists():
                (temp_workspace / "lean-toolchain").write_text(
                    candidate.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                break

        return temp_workspace, "Dataset"


class RocqTrainingPipelineExtractor:
    """Rocq extraction with diagnostics-based Print/Check querying (training step_0 style)."""

    def __init__(
        self,
        benchmark_root: Path,
        timeout: int = 120,
        stem_retries: int = 4,
        retry_sleep_sec: float = 2.0,
    ):
        self.benchmark_root = benchmark_root
        self.rocq_dir = benchmark_root / "rocq"
        self.timeout = timeout
        self.stem_retries = max(1, int(stem_retries))
        self.retry_sleep_sec = max(0.0, float(retry_sleep_sec))

    def _decl_names_by_stem(self, stems: Sequence[str]) -> Dict[str, List[str]]:
        names: Dict[str, List[str]] = {}
        for stem in stems:
            decls = parse_rocq_declarations(self.rocq_dir / f"{stem}.v")
            names[stem] = [decl.name for decl in decls]
        return names

    @staticmethod
    def _module_prefixes(file_path: Path) -> List[str]:
        prefixes: List[str] = []
        module_re = re.compile(r"^\s*Module\s+([A-Za-z_][A-Za-z0-9_']*)\s*\.")
        for line in file_path.read_text(encoding="utf-8").splitlines():
            match = module_re.match(line)
            if not match:
                continue
            name = match.group(1).strip()
            if name and name not in prefixes:
                prefixes.append(name)
        return prefixes

    @staticmethod
    def _is_rocq_error_message(text: str) -> bool:
        msg = (text or "").strip().lower()
        if not msg:
            return True
        return (
            msg.startswith("error:")
            or " was not found " in msg
            or "not a defined object" in msg
            or "cannot find a physical path bound to logical path" in msg
        )

    @classmethod
    def _theorem_query_candidates(cls, theorem_name: str, module_prefixes: List[str]) -> List[str]:
        # Prefer qualified names first: many benchmark files wrap declarations in a Module.
        candidates: List[str] = []
        for prefix in module_prefixes:
            cand = f"{prefix}.{theorem_name}"
            if cand not in candidates:
                candidates.append(cand)
        if theorem_name not in candidates:
            candidates.append(theorem_name)
        return candidates

    @staticmethod
    def _fallback_dep_candidates_from_term(
        term: str,
        theorem_name: str,
        selected_name: str,
    ) -> List[str]:
        rhs = term.split("=", maxsplit=1)[1] if "=" in term else term
        candidates: List[str] = []
        seen: Set[str] = set()
        short_theorem = theorem_name.split(".")[-1]
        short_selected = selected_name.split(".")[-1]

        for token in _ROCQ_TERM_IDENT_RE.findall(rhs):
            tok = token.strip()
            if not tok or tok in seen:
                continue
            seen.add(tok)

            short = tok.split(".")[-1]
            if (
                tok in _ROCQ_RESERVED_IDENTIFIERS
                or short in _ROCQ_RESERVED_IDENTIFIERS
                or tok == theorem_name
                or tok == selected_name
                or short == short_theorem
                or short == short_selected
            ):
                continue

            # Heuristic to avoid obvious local binders.
            if (
                "." not in tok
                and "_" not in tok
                and tok[:1].islower()
                and len(tok) <= 3
            ):
                continue
            # Keep only likely global references.
            if "." not in tok and "_" not in tok and not tok[:1].isupper():
                continue

            candidates.append(tok)
            if len(candidates) >= 64:
                break

        return candidates

    @staticmethod
    def _parse_dependency_names(raw: str) -> List[str]:
        names: List[str] = []
        seen: Set[str] = set()

        for line in raw.splitlines():
            text = line.strip()
            if not text:
                continue
            if any(skip in text for skip in _ROCQ_SKIP_LINES):
                continue
            if text in {"Transparent constants:", "Opaque constants:"}:
                continue

            if ":" not in text:
                continue

            dep = text.split(":", maxsplit=1)[0].strip()
            if dep in _ROCQ_RESERVED_IDENTIFIERS:
                continue
            if dep and _ROCQ_IDENT_RE.match(dep) and dep not in seen:
                seen.add(dep)
                names.append(dep)

        return names

    @staticmethod
    def _parse_check_type(dep: str, check_output: str) -> str:
        text = check_output.strip()
        prefix = f"{dep} :"
        idx = text.find(prefix)
        if idx >= 0:
            return text[idx + len(prefix) :].strip()
        return text

    def _extract_stem_with_retries(
        self,
        stem: str,
        theorem_names: List[str],
    ) -> List[TheoremArtifact]:
        last_error: Exception | None = None
        for attempt in range(1, self.stem_retries + 1):
            module_prefixes = self._module_prefixes(self.rocq_dir / f"{stem}.v")
            session = RocqLspQuerySession(
                file_path=self.rocq_dir / f"{stem}.v",
                workspace=self.rocq_dir.resolve(),
                timeout=self.timeout,
            )
            try:
                stem_artifacts: List[TheoremArtifact] = []
                for theorem_name in theorem_names:
                    term_raw = None
                    selected_name = None
                    tried: List[str] = []
                    for candidate in self._theorem_query_candidates(theorem_name, module_prefixes):
                        tried.append(candidate)
                        try:
                            candidate_raw = session.query(
                                "Print",
                                candidate,
                                f"Print {candidate}.",
                                timeout_sec=max(10.0, min(float(self.timeout), 60.0)),
                            )
                        except ExtractionError:
                            continue
                        if self._is_rocq_error_message(candidate_raw):
                            continue
                        term_raw = candidate_raw
                        selected_name = candidate
                        break
                    if term_raw is None or selected_name is None:
                        raise ExtractionError(
                            f"Rocq theorem print failed for {stem}:{theorem_name}. "
                            f"Tried: {', '.join(tried)}"
                        )
                    term = (term_raw or "").strip()
                    if not term:
                        raise ExtractionError(
                            f"Rocq theorem has empty proof term: {stem}:{theorem_name}"
                        )

                    deps_raw = session.query(
                        "Print All Dependencies",
                        selected_name,
                        f"Print All Dependencies {selected_name}.",
                        timeout_sec=max(10.0, min(float(self.timeout), 60.0)),
                    )
                    dep_names = [
                        dep
                        for dep in self._parse_dependency_names(deps_raw)
                        if dep != theorem_name
                    ]
                    if not dep_names:
                        dep_names = self._fallback_dep_candidates_from_term(
                            term=term,
                            theorem_name=theorem_name,
                            selected_name=selected_name,
                        )

                    dependencies: List[DependencyInfo] = []
                    seen: Set[str] = set()
                    for dep in dep_names:
                        if dep in seen:
                            continue
                        seen.add(dep)

                        try:
                            check_raw = session.query(
                                "Check",
                                dep,
                                f"Check {dep}.",
                                timeout_sec=min(1.0, float(self.timeout)),
                            )
                        except ExtractionError:
                            # Ignore non-global identifiers reported by dependency listing.
                            continue
                        dep_type = self._parse_check_type(dep, check_raw)
                        dependencies.append(
                            DependencyInfo(name=dep, fqn=dep, type=dep_type)
                        )

                    stem_artifacts.append(
                        TheoremArtifact(
                            source_stem=stem,
                            name=theorem_name,
                            term=term,
                            dependencies=dependencies,
                        )
                    )

                return stem_artifacts
            except ExtractionError as exc:
                last_error = exc
                if attempt >= self.stem_retries:
                    break
                sleep_s = self.retry_sleep_sec * attempt
                logger.warning(
                    "Rocq extraction transient failure for stem=%s (attempt %d/%d): %s. Retrying in %.1fs",
                    stem,
                    attempt,
                    self.stem_retries,
                    exc,
                    sleep_s,
                )
                print(
                    f"[extract][rocq] transient failure stem={stem} attempt={attempt}/{self.stem_retries}: {exc}. "
                    f"retry_in={sleep_s:.1f}s",
                    flush=True,
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)
            finally:
                session.close()

        raise ExtractionError(
            f"Rocq extraction failed for {stem} after {self.stem_retries} attempts: {last_error}"
        )

    def extract(
        self,
        stems: Sequence[str],
        on_stem_done: Callable[[str, List[TheoremArtifact]], None] | None = None,
    ) -> List[TheoremArtifact]:
        decls_by_stem = self._decl_names_by_stem(stems)
        artifacts: List[TheoremArtifact] = []

        for idx, stem in enumerate(stems, start=1):
            print(f"[extract][rocq] {idx}/{len(stems)} stem={stem}", flush=True)
            stem_artifacts = self._extract_stem_with_retries(stem, decls_by_stem[stem])
            artifacts.extend(stem_artifacts)
            if on_stem_done is not None:
                on_stem_done(stem, stem_artifacts)

        return artifacts


class BenchmarkProofTermExtractionJob:
    def __init__(
        self,
        benchmark_root: Path,
        output_dir: Path,
        rocq_timeout: int,
        rocq_stem_retries: int = 4,
        rocq_stem_retry_sleep: float = 2.0,
        lean_stem_retries: int = 4,
        lean_stem_retry_sleep: float = 2.0,
    ):
        self.benchmark_root = benchmark_root
        self.output_dir = output_dir
        self.rocq_timeout = rocq_timeout
        self.rocq_stem_retries = max(1, int(rocq_stem_retries))
        self.rocq_stem_retry_sleep = max(0.0, float(rocq_stem_retry_sleep))
        self.lean_stem_retries = max(1, int(lean_stem_retries))
        self.lean_stem_retry_sleep = max(0.0, float(lean_stem_retry_sleep))
        self.progress_path = self.output_dir / ".extract_progress.json"
        self.lean_cache_path = self.output_dir / ".lean.partial.json"
        self.rocq_cache_path = self.output_dir / ".rocq.partial.json"

    def _common_stems(self) -> List[str]:
        lean_dir = get_workspace("lean", benchmark_root=self.benchmark_root)
        rocq_dir = get_workspace("rocq", benchmark_root=self.benchmark_root)
        lean_stems = {path.stem for path in lean_dir.glob("*.lean")}
        rocq_stems = {path.stem for path in rocq_dir.glob("*.v")}
        return sorted(lean_stems.intersection(rocq_stems))

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _pair_set(artifacts: Sequence[TheoremArtifact]) -> Set[Tuple[str, str]]:
        return {(entry.source_stem, entry.name) for entry in artifacts}

    @staticmethod
    def _artifact_key(entry: TheoremArtifact) -> Tuple[str, str]:
        return (entry.source_stem, entry.name)

    def _load_progress(self) -> dict:
        if not self.progress_path.exists():
            return {"lean_completed_stems": [], "rocq_completed_stems": []}
        payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
        lean_completed = payload.get("lean_completed_stems", [])
        rocq_completed = payload.get("rocq_completed_stems", [])
        if not isinstance(lean_completed, list):
            lean_completed = []
        if not isinstance(rocq_completed, list):
            rocq_completed = []
        return {
            "lean_completed_stems": [str(x) for x in lean_completed],
            "rocq_completed_stems": [str(x) for x in rocq_completed],
        }

    def _write_progress(self, progress: dict) -> None:
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.write_text(
            json.dumps(progress, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_partial_artifacts(self, path: Path) -> Dict[Tuple[str, str], TheoremArtifact]:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return {}
        out: Dict[Tuple[str, str], TheoremArtifact] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            stem = str(row.get("source_stem", "")).strip()
            name = str(row.get("name", "")).strip()
            term = str(row.get("term", "")).strip()
            if not stem or not name or not term:
                continue
            deps_raw = row.get("dependencies", [])
            deps: List[DependencyInfo] = []
            if isinstance(deps_raw, list):
                for dep in deps_raw:
                    if not isinstance(dep, dict):
                        continue
                    dep_name = str(dep.get("name", "")).strip()
                    dep_fqn = str(dep.get("fqn", "")).strip()
                    dep_type = str(dep.get("type", "")).strip()
                    if not dep_name or not dep_fqn:
                        continue
                    deps.append(DependencyInfo(name=dep_name, fqn=dep_fqn, type=dep_type))
            artifact = TheoremArtifact(source_stem=stem, name=name, term=term, dependencies=deps)
            out[self._artifact_key(artifact)] = artifact
        return out

    def _write_partial_artifacts(
        self,
        path: Path,
        artifacts: Dict[Tuple[str, str], TheoremArtifact],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            artifact.to_json()
            for artifact in sorted(artifacts.values(), key=lambda e: (e.source_stem, e.name))
        ]
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    def run(
        self,
        stems_override: Sequence[str] | None = None,
        max_stems: int | None = None,
    ) -> dict:
        stems = self._common_stems()
        if stems_override:
            keep = set(stems_override)
            stems = [stem for stem in stems if stem in keep]
        if max_stems is not None:
            stems = stems[: max(0, int(max_stems))]

        if not stems:
            raise ExtractionError(
                f"No paired Lean/Rocq files found under {self.benchmark_root}."
            )

        progress = self._load_progress()
        lean_cache = self._load_partial_artifacts(self.lean_cache_path)
        rocq_cache = self._load_partial_artifacts(self.rocq_cache_path)

        lean_completed = set(progress["lean_completed_stems"])
        rocq_completed = set(progress["rocq_completed_stems"])

        lean_pending = [stem for stem in stems if stem not in lean_completed]
        rocq_pending = [stem for stem in stems if stem not in rocq_completed]

        def on_lean_stem_done(stem: str, stem_artifacts: List[TheoremArtifact]) -> None:
            for artifact in stem_artifacts:
                lean_cache[self._artifact_key(artifact)] = artifact
            lean_completed.add(stem)
            progress["lean_completed_stems"] = sorted(lean_completed)
            self._write_partial_artifacts(self.lean_cache_path, lean_cache)
            self._write_progress(progress)

        def on_rocq_stem_done(stem: str, stem_artifacts: List[TheoremArtifact]) -> None:
            for artifact in stem_artifacts:
                rocq_cache[self._artifact_key(artifact)] = artifact
            rocq_completed.add(stem)
            progress["rocq_completed_stems"] = sorted(rocq_completed)
            self._write_partial_artifacts(self.rocq_cache_path, rocq_cache)
            self._write_progress(progress)

        lean_new = LeanTrainingPipelineExtractor(
            self.benchmark_root,
            stem_retries=self.lean_stem_retries,
            retry_sleep_sec=self.lean_stem_retry_sleep,
        ).extract(lean_pending, on_stem_done=on_lean_stem_done)
        for artifact in lean_new:
            lean_cache[self._artifact_key(artifact)] = artifact

        rocq_new = RocqTrainingPipelineExtractor(
            self.benchmark_root,
            timeout=self.rocq_timeout,
            stem_retries=self.rocq_stem_retries,
            retry_sleep_sec=self.rocq_stem_retry_sleep,
        ).extract(rocq_pending, on_stem_done=on_rocq_stem_done)
        for artifact in rocq_new:
            rocq_cache[self._artifact_key(artifact)] = artifact

        lean_artifacts = sorted(lean_cache.values(), key=lambda e: (e.source_stem, e.name))
        rocq_artifacts = sorted(rocq_cache.values(), key=lambda e: (e.source_stem, e.name))

        lean_pairs = self._pair_set(lean_artifacts)
        rocq_pairs = self._pair_set(rocq_artifacts)
        missing_in_lean = sorted(rocq_pairs.difference(lean_pairs))
        missing_in_rocq = sorted(lean_pairs.difference(rocq_pairs))
        if missing_in_lean or missing_in_rocq:
            raise ExtractionError(
                "Lean/Rocq extraction mismatch. "
                f"Missing in lean: {missing_in_lean[:5]} "
                f"Missing in rocq: {missing_in_rocq[:5]}"
            )

        lean_rows = [entry.to_json() for entry in lean_artifacts]
        rocq_rows = [entry.to_json() for entry in rocq_artifacts]

        self._write_json(self.output_dir / "lean.json", lean_rows)
        self._write_json(self.output_dir / "rocq.json", rocq_rows)
        # Clean partial state once the full extraction succeeds.
        if self.progress_path.exists():
            self.progress_path.unlink()
        if self.lean_cache_path.exists():
            self.lean_cache_path.unlink()
        if self.rocq_cache_path.exists():
            self.rocq_cache_path.unlink()

        manifest = {
            "benchmark_root": str(self.benchmark_root),
            "output_dir": str(self.output_dir),
            "stems": stems,
            "counts": {"lean": len(lean_rows), "rocq": len(rocq_rows)},
            "pipeline": {
                "lean": "src.lean_rocq_translation.step_4.exec::TheoremExtractor",
                "rocq": "coq-lsp diagnostics query (Print / Print All Dependencies / Check)",
            },
            "retries": {
                "lean_stem_retries": self.lean_stem_retries,
                "lean_stem_retry_sleep": self.lean_stem_retry_sleep,
                "rocq_stem_retries": self.rocq_stem_retries,
                "rocq_stem_retry_sleep": self.rocq_stem_retry_sleep,
            },
        }
        self._write_json(self.output_dir / "manifest.json", manifest)

        return {
            "output_dir": str(self.output_dir),
            "lean_rows": len(lean_rows),
            "rocq_rows": len(rocq_rows),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract canonical proof terms plus dependency/type metadata for new_benchmark "
            "using the same extraction family as training dataset generation."
        )
    )
    parser.add_argument("--benchmark-root", default="new_benchmark")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <benchmark-root>/proof_terms",
    )
    parser.add_argument("--rocq-timeout", type=int, default=120)
    parser.add_argument(
        "--rocq-stem-retries",
        type=int,
        default=4,
        help="Per-stem retries for Rocq extraction on transient diagnostics issues.",
    )
    parser.add_argument(
        "--rocq-stem-retry-sleep",
        type=float,
        default=2.0,
        help="Base sleep (seconds) between Rocq stem retries (linear backoff).",
    )
    parser.add_argument(
        "--lean-stem-retries",
        type=int,
        default=4,
        help="Per-stem retries for Lean extraction on transient empty/missing results.",
    )
    parser.add_argument(
        "--lean-stem-retry-sleep",
        type=float,
        default=2.0,
        help="Base sleep (seconds) between Lean stem retries (linear backoff).",
    )
    parser.add_argument(
        "--stems",
        default=None,
        help="Optional comma-separated subset of source stems to extract.",
    )
    parser.add_argument(
        "--max-stems",
        type=int,
        default=None,
        help="Optional cap on number of stems (after --stems filtering).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    benchmark_root = Path(args.benchmark_root)
    output_dir = Path(args.output_dir) if args.output_dir else benchmark_root / "proof_terms"

    stems_override = None
    if args.stems:
        stems_override = [item.strip() for item in str(args.stems).split(",") if item.strip()]

    result = BenchmarkProofTermExtractionJob(
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        rocq_timeout=args.rocq_timeout,
        rocq_stem_retries=args.rocq_stem_retries,
        rocq_stem_retry_sleep=args.rocq_stem_retry_sleep,
        lean_stem_retries=args.lean_stem_retries,
        lean_stem_retry_sleep=args.lean_stem_retry_sleep,
    ).run(stems_override=stems_override, max_stems=args.max_stems)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
