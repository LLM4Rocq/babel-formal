from __future__ import annotations

import json
import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict

import leanclient as lc


@dataclass
class Range:
    start_line: int
    start_char: int
    end_line: int
    end_char: int

    @staticmethod
    def from_lsp(r: dict) -> "Range":
        return Range(
            start_line=r["start"]["line"],
            start_char=r["start"]["character"],
            end_line=r["end"]["line"],
            end_char=r["end"]["character"],
        )

    def contains(self, line: int, char: int) -> bool:
        if (line < self.start_line) or (line > self.end_line):
            return False
        if line == self.start_line and char < self.start_char:
            return False
        if line == self.end_line and char > self.end_char:
            return False
        return True


@dataclass
class SymbolUse:
    name: str
    type: Optional[str]
    file: Optional[str]
    line: Optional[int]
    character: Optional[int]
    category: str
    origin: str  # "statement" | "proof"
    check_text: Optional[str] = None   # pretty-printed type (aka #check)
    print_text: Optional[str] = None   # declaration pretty-print (aka #print)


@dataclass
class TheoremInfo:
    name: str
    range_full: Range
    proof_range: Range
    header_text: str
    statement_text: str
    proof_text: str

HEADER_SPLIT_RE = re.compile(r"\b(:=|by)\b")
THEOREM_LINE_RE = re.compile(r"^(\s*)(theorem|lemma)\s+([A-Za-z0-9_\.]+)")
NEXT_TOPLVL_RE = re.compile(r"^(\s*)(theorem|lemma|def|axiom|structure|class|inductive|namespace|end)\b")

HOVER_FQN_RE = re.compile(r"```lean\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:[({\s]|$)")


def _split_header_proof(file_text: str, r: Range) -> Tuple[str, str]:
    lines = file_text.splitlines()
    slice_lines = lines[r.start_line : r.end_line + 1]
    if not slice_lines:
        return "", ""
    slice_lines[0] = slice_lines[0][r.start_char :]
    slice_lines[-1] = slice_lines[-1][: r.end_char]
    decl_text = "\n".join(slice_lines)

    m = HEADER_SPLIT_RE.search(decl_text)
    if not m:
        return decl_text, ""
    i = m.start()
    return decl_text[:i], decl_text[i:]


class LSPNavigator:
    def __init__(self, client: lc.LeanLSPClient, file_path: str, project_root: Optional[str] = None):
        self.client = client
        self.project_root = Path(project_root or getattr(client, "project_path", ".")).resolve()
        self._opened: set[str] = set()

        # Always store and use the *local* (project-relative) path with POSIX slashes.
        self.file_path = self._to_local(file_path)

        self.sfc = client.create_file_client(self.file_path)
        self.sfc.open_file()
        self._opened.add(self.file_path)
        self.file_text = self.sfc.get_file_content()

    def _to_local(self, p: str) -> str:
        """Convert absolute or relative filesystem path to project-relative POSIX path."""
        pp = Path(p)
        if pp.is_absolute():
            try:
                pp = pp.resolve().relative_to(self.project_root)
            except ValueError:
                # If it's outside the project_root, leave it absolute but still POSIX.
                pp = pp.resolve()
        else:
            pp = (self.project_root / pp).resolve().relative_to(self.project_root)
        return pp.as_posix()

    def _ensure_open(self, path: str, timeout: float = 180.0) -> None:
        lp = self._to_local(path)  # normalize to project-relative
        if lp in self._opened:
            return
        sfc_other = self.client.create_file_client(lp)
        # first try (large mathlib modules can be slow)
        try:
            sfc_other.open_file(timeout=timeout)
        except FileNotFoundError:
            # likely just slow diagnostics; quick backoff and retry longer
            time.sleep(0.5)
            sfc_other.open_file(timeout=timeout * 2)
        self._opened.add(lp)

    def document_symbols(self) -> List[dict]:
        return self.sfc.get_document_symbols() or []

    def hover(self, line: int, character: int) -> Optional[dict]:
        return self.sfc.get_hover(line=line, character=character)

    def definitions(self, line: int, character: int) -> List[dict]:
        return self.sfc.get_definitions(line=line, character=character) or []

    def _pos_end_of_file(self, text: str) -> tuple[int, int]:
        if not text:
            return 0, 0
        lines = text.splitlines()
        if text.endswith("\n"):
            return len(lines), 0
        else:
            return len(lines) - 1, (len(lines[-1]) if lines else 0)

    def _advance_pos(self, start_line: int, start_char: int, inserted: str) -> tuple[int, int]:
        segs = inserted.splitlines()
        if "\n" not in inserted:
            return start_line, start_char + len(inserted)
        return start_line + (len(segs) - 1), len(segs[-1])

    def resolve_fqn_via_definition(self, defs: List[dict]) -> Optional[str]:
        if not defs:
            return None
        d0 = defs[0]
        uri = d0.get("uri")
        r = (d0.get("range") or {}).get("start") or {}
        if not uri or not uri.startswith("file://"):
            return None

        # Turn the file:// URI into a local (project-relative) path
        target_abs = uri[len("file://"):]      # absolute
        target_local = self._to_local(target_abs)

        self._ensure_open(target_local)

        # IMPORTANT: ask for declarations with the *same* local path
        decls = self.client.get_declarations(
            target_local, r.get("line", 0), r.get("character", 0)
        ) or []

        for dec in decls:
            s = (dec.get("range") or {}).get("start") or {}
            if s.get("line") == r.get("line") and s.get("character") == r.get("character"):
                nm = dec.get("name")
                if nm:
                    return nm
        for dec in decls:
            nm = dec.get("name")
            if nm:
                return nm
        return None

    def _collect_diagnostics_with_poll(self, attempts: int = 6, delay: float = 0.05) -> List[dict]:
        """
        Diagnostics may arrive just after update_file returns; poll briefly.
        """
        diags = self.sfc.get_diagnostics() or []
        if diags:
            return diags
        for _ in range(attempts - 1):
            time.sleep(delay)
            diags = self.sfc.get_diagnostics() or []
            if diags:
                break
        return diags

    def print_decl_via_diagnostics(self, fqn: str, timeout: float = 120.0) -> Optional[str]:
        if not fqn or "." not in fqn:
            return None

        # Ensure the *main* file is open under its local path
        if self.file_path not in self._opened:
            self._ensure_open(self.file_path)

        # Grab current content; reopen if the file manager lost it
        try:
            original = self.sfc.get_file_content()
        except FileNotFoundError:
            # Reopen with the same local path and try again
            self.sfc.open_file(timeout=timeout)
            original = self.sfc.get_file_content()

        start_line, start_char = self._pos_end_of_file(original)

        marker = f"-- __PRINT_MARKER__ {fqn}"
        patch = ("" if (not original or original.endswith("\n")) else "\n") + f"{marker}\n#print {fqn}\n"
        end_line, end_char = self._advance_pos(start_line, start_char, patch)

        append_change = lc.DocumentContentChange(
            text=patch,
            start=[start_line, start_char],
            end=[start_line, start_char],
        )
        self.sfc.update_file(changes=[append_change], timeout=timeout)

        diags = self._collect_diagnostics_with_poll()

        def in_appended_range(d: dict) -> bool:
            rng = d.get("range") or {}
            s = rng.get("start") or {}
            e = rng.get("end") or {}
            sl, sc = s.get("line", -1), s.get("character", -1)
            el, ec = e.get("line", -1), e.get("character", -1)
            if sl < start_line or (sl == start_line and sc < start_char):
                return False
            if el > end_line or (el == end_line and ec > end_char):
                return False
            return True

        msg = None
        for d in diags:
            if d.get("severity", 3) in (3, 4) and in_appended_range(d):
                m = d.get("message", "")
                if m.strip():
                    msg = m
                    break
        if msg is None:
            for d in diags:
                if d.get("severity", 3) in (3, 4):
                    m = d.get("message", "")
                    if fqn in (m or ""):
                        msg = m
                        break

        # Clean up our appended text
        delete_change = lc.DocumentContentChange(
            text="",
            start=[start_line, start_char],
            end=[end_line, end_char],
        )
        self.sfc.update_file(changes=[delete_change], timeout=timeout)

        # Refresh local cache
        try:
            self.file_text = self.sfc.get_file_content()
        except FileNotFoundError:
            # If Lean evicted it (rare), reopen to keep navigator consistent
            self.sfc.open_file(timeout=timeout)
            self.file_text = self.sfc.get_file_content()

        return msg


class TheoremExtractor:
    def __init__(self, project_path: str):
        self.client = lc.LeanLSPClient(project_path)

    def close(self):
        self.client.close()

    # --- Robust discovery ---
    def _scan_text_for_theorems(self, file_text: str) -> List[Tuple[int, int, str]]:
        out: List[Tuple[int, int, str]] = []
        for i, line in enumerate(file_text.splitlines()):
            m = THEOREM_LINE_RE.match(line)
            if m:
                name = m.group(3)
                char = m.start(2)  # start of keyword
                out.append((i, char, name))
        return out

    def _fallback_range(self, file_text: str, start_line: int) -> Range:
        lines = file_text.splitlines()
        end_line = start_line
        for j in range(start_line + 1, len(lines)):
            if NEXT_TOPLVL_RE.match(lines[j]):
                end_line = j - 1
                break
        else:
            end_line = len(lines) - 1
        return Range(start_line=start_line, start_char=0, end_line=end_line, end_char=len(lines[end_line]))

    def _find_theorems(self, nav: LSPNavigator) -> List[dict]:
        syms = nav.document_symbols()
        sym_by_name = {}
        for s in syms:
            nm = s.get("name")
            if nm:
                sym_by_name.setdefault(nm, []).append(s)
        for lst in sym_by_name.values():
            lst.sort(key=lambda s: s.get("range", {}).get("start", {}).get("line", 10**9))

        theorems: List[dict] = []
        for (ln, ch, nm) in self._scan_text_for_theorems(nav.file_text):
            cand = None
            if nm in sym_by_name:
                for s in sym_by_name[nm]:
                    sline = s.get("range", {}).get("start", {}).get("line", -1)
                    if sline >= ln - 2:
                        cand = s
                        break
            if cand is None:
                r = self._fallback_range(nav.file_text, ln)
                cand = {
                    "name": nm,
                    "range": {
                        "start": {"line": r.start_line, "character": r.start_char},
                        "end": {"line": r.end_line, "character": r.end_char},
                    },
                    "selectionRange": {
                        "start": {"line": ln, "character": ch},
                        "end": {"line": ln, "character": ch + len("theorem")},
                    },
                }
            cand["kind"] = "theorem" if "theorem" in nav.file_text.splitlines()[ln] else "lemma"
            theorems.append(cand)

        theorems.sort(key=lambda d: (
            d.get("range", {}).get("start", {}).get("line", 10**9),
            d.get("range", {}).get("start", {}).get("character", 10**9),
        ))
        return theorems


    def extract_file(self, file_path: str) -> List[TheoremInfo]:
        nav = LSPNavigator(self.client, file_path)
        theorems = self._find_theorems(nav)
        results: List[TheoremInfo] = []
        for th in theorems:
            name = th.get("name", "<anonymous>")
            r = Range.from_lsp(th.get("range"))
            header, proof = _split_header_proof(nav.file_text, r)

            proof_range = r
            if proof.strip():
                pre_lines = header.count("\n")
                last_line_len = len(header.split("\n")[-1]) if header else 0
                proof_start_line = r.start_line + pre_lines
                proof_start_char = (0 if pre_lines > 0 else r.start_char) + last_line_len
                proof_range = Range(proof_start_line, proof_start_char, r.end_line, r.end_char)

            results.append(TheoremInfo(
                name=name,
                range_full=r,
                proof_range=proof_range,
                header_text=header.strip(),
                statement_text=header.strip(),
                proof_text=proof.strip(),
            ))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mathlib", help='Mathlib path')
    parser.add_argument("--output", default='export/lean_dataset/step_1')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    extractor = TheoremExtractor(args.mathlib)
    to_do = []
    for root, _, files in os.walk(os.path.join(args.mathlib, 'Mathlib')):
        
        for file in files:
            if file.endswith('.lean'):
                origin = os.path.relpath(root, args.mathlib)
                filepath = os.path.join(origin, file)

                with open(os.path.join(root, file), 'r') as file:
                    content = file.read()
                
                if content.count('\ntheorem') > 1:
                    to_do.append(filepath)
    
    data = defaultdict(list)
    for filepath in tqdm(to_do):
        infos = extractor.extract_file(filepath)
        data = {filepath: []}
        for th in infos:
            data[filepath].append({
                "name": th.name,
                "header": th.header_text,
                "statement": th.statement_text,
                "proof": th.proof_text
            })

        name_file = filepath.replace('/', '_').replace('.lean', '')
        with open(os.path.join(args.output, name_file), 'w') as file:
            json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()
