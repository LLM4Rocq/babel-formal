from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from tqdm import tqdm
import leanclient as lc

"""
Fourth step: Extract Lean theorems with symbols and canonical proof terms (#print) by resolving fully-qualified names.
"""

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
    header_text: str
    statement_text: str
    proof_text: str
    symbols: List[SymbolUse]
    # NEW: pretty-printed proof term (from `#print name`, slice after ':=')
    term: Optional[str] = None


HEADER_SPLIT_RE = re.compile(r"\b(:=|by)\b")
THEOREM_LINE_RE = re.compile(r"^(\s*)(theorem|lemma)\s+([A-Za-z0-9_\.]+)")
NEXT_TOPLVL_RE = re.compile(r"^(\s*)(theorem|lemma|def|axiom|structure|class|inductive|namespace|end)\b")

# Parse an FQN out of Lean hover code blocks like:
# ```lean
# List.map {α : Type} ...
# ```
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


def _first_identifier_positions(text: str) -> Iterable[Tuple[int, int, str]]:
    ident = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*")
    for i, line in enumerate(text.splitlines()):
        for m in ident.finditer(line):
            yield i, m.start(), m.group(0)


def _hover_to_type(hover: Optional[dict]) -> Optional[str]:
    if not hover:
        return None
    contents = hover.get("contents") if isinstance(hover, dict) else None
    if not contents:
        return None
    if isinstance(contents, dict):
        val = contents.get("value") or contents.get("valueRendered")
    elif isinstance(contents, list) and contents:
        val = contents[0].get("value") if isinstance(contents[0], dict) else None
    else:
        val = None
    return val


def _fqn_from_hover_text(hover_text: Optional[str]) -> Optional[str]:
    if not hover_text:
        return None
    m = HOVER_FQN_RE.search(hover_text)
    if m:
        return m.group(1)
    return None


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
        if not fqn:
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
        """
        Return tuples (line, name_start_char, name).
        We anchor on the *name* so we can resolve FQNs via definitions/hover.
        """
        out: List[Tuple[int, int, str]] = []
        for i, line in enumerate(file_text.splitlines()):
            m = THEOREM_LINE_RE.match(line)
            if m:
                name = m.group(3)
                char_name = m.start(3)  # start of theorem/lemma name
                out.append((i, char_name, name))
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
        for (ln, name_ch, nm) in self._scan_text_for_theorems(nav.file_text):
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
                    # IMPORTANT: selectionRange now spans the NAME, not the keyword
                    "selectionRange": {
                        "start": {"line": ln, "character": name_ch},
                        "end": {"line": ln, "character": name_ch + len(nm)},
                    },
                }
            # label kind
            cand["kind"] = "theorem" if "theorem" in nav.file_text.splitlines()[ln] else "lemma"
            # ensure selectionRange is set to the name even when we used a symbol hit
            if "selectionRange" not in cand or not cand["selectionRange"]:
                cand["selectionRange"] = {
                    "start": {"line": ln, "character": name_ch},
                    "end": {"line": ln, "character": name_ch + len(nm)},
                }
            theorems.append(cand)

        theorems.sort(key=lambda d: (
            d.get("range", {}).get("start", {}).get("line", 10**9),
            d.get("range", {}).get("start", {}).get("character", 10**9),
        ))
        return theorems

    def _collect_symbols_from_range(self, nav: LSPNavigator, r: Range, origin: str) -> List[SymbolUse]:
        uses: List[SymbolUse] = []
        lines = nav.file_text.splitlines()
        slice_lines = lines[r.start_line : r.end_line + 1]
        if not slice_lines:
            return uses
        slice_lines[0] = slice_lines[0][r.start_char:]
        slice_lines[-1] = slice_lines[-1][:r.end_char]
        text = "\n".join(slice_lines)

        try:
            for l_off, c, name in _first_identifier_positions(text):
                abs_line = r.start_line + l_off
                abs_col = c if l_off > 0 else (r.start_char + c)

                h = nav.hover(abs_line, abs_col)
                check_text = _hover_to_type(h)

                defs = nav.definitions(abs_line, abs_col)

                fqn = nav.resolve_fqn_via_definition(defs)
                if not fqn:
                    fqn = _fqn_from_hover_text(check_text)

                print_text = None
                if fqn:
                    print_text = nav.print_decl_via_diagnostics(fqn)

                def_file = def_line = def_char = None
                if defs:
                    d0 = defs[0]
                    target_uri = d0.get("uri")
                    if target_uri and target_uri.startswith("file://"):
                        def_file = target_uri[len("file://"):]
                    if d0.get("range"):
                        dr = d0["range"]["start"]
                        def_line, def_char = dr.get("line"), dr.get("character")
                
                category=""
                uses.append(SymbolUse(
                    name=name,
                    category=category,
                    type=check_text,
                    file=def_file,
                    line=def_line,
                    character=def_char,
                    origin=origin,
                    check_text=check_text,
                    print_text=print_text,
                ))
        except Exception as e:
            print(e)

        return uses

    # NEW: resolve the fully-qualified name of a theorem given the token position at its name
    def _resolve_theorem_fqn(self, nav: LSPNavigator, line: int, ch: int) -> Optional[str]:
        defs = nav.definitions(line, ch)
        fqn = nav.resolve_fqn_via_definition(defs)
        if fqn:
            return fqn
        # fallback via hover
        h = nav.hover(line, ch)
        return _fqn_from_hover_text(_hover_to_type(h))

    def extract_file(self, file_path: str, filtered_list: set) -> List[TheoremInfo]:
        nav = LSPNavigator(self.client, file_path)
        theorems = self._find_theorems(nav)
        results: List[TheoremInfo] = []
        for th in theorems:
            try:
                name = th.get("name", "<anonymous>")

                if name not in filtered_list:
                    continue
                r = Range.from_lsp(th.get("range"))
                header, proof = _split_header_proof(nav.file_text, r)

                header_lines = header.splitlines()
                header_end_char = (len(header_lines[-1]) if header_lines else r.start_char)
                header_range = Range(r.start_line, r.start_char,
                                    r.start_line + header.count("\n"),
                                    header_end_char)

                proof_range = r
                if proof.strip():
                    pre_lines = header.count("\n")
                    last_line_len = len(header.split("\n")[-1]) if header else 0
                    proof_start_line = r.start_line + pre_lines
                    proof_start_char = (0 if pre_lines > 0 else r.start_char) + last_line_len
                    proof_range = Range(proof_start_line, proof_start_char, r.end_line, r.end_char)

                header_symbols = self._collect_symbols_from_range(nav, header_range, origin="statement")
                proof_symbols: List[SymbolUse] = []
                if proof.strip():
                    proof_symbols = self._collect_symbols_from_range(nav, proof_range, origin="proof")

                # merge de-duping
                seen = set()
                merged: List[SymbolUse] = []
                for su in header_symbols + proof_symbols:
                    if su.name in seen:
                        continue
                    seen.add(su.name)
                    merged.append(su)
                sr = th.get("selectionRange", {}).get("start", {})  # this points at the NAME now
                fqn = None

                if "line" in sr and "character" in sr:
                    fqn = self._resolve_theorem_fqn(nav, sr["line"], sr["character"])
                term = None
                if fqn:
                    term = nav.print_decl_via_diagnostics(fqn)

                results.append(TheoremInfo(
                    name=name,
                    range_full=r,
                    header_text=header.strip(),
                    statement_text=header.strip(),
                    proof_text=proof.strip(),
                    symbols=merged,
                    term=term,  # NEW
                ))
            except Exception as e:
                print(e)
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default='/home/theo/Documents/github/babel-formal/dataset')
    ap.add_argument("--input", default='export_dataset_lean')
    args = ap.parse_args()

    extractor = TheoremExtractor(args.workspace)

    for filename in os.listdir(args.input):
        filepath = os.path.join(args.input, filename)
        with open(filepath, 'r') as file:
            content = json.load(file)
        for source in content:
            try:
                filename_normalized = filename.replace('/', '_')
                export_path = os.path.join("export_step_4_dataset", f"{filename_normalized}")
                if os.path.exists(export_path):
                    continue
                filtered_list = set([entry['name'] for entry in content[source]])
                infos = extractor.extract_file(source+ '.lean', filtered_list)
                data = []
                for th in infos:
                    data.append({
                        "name": th.name,
                        "range": asdict(th.range_full),
                        "header": th.header_text,
                        "statement": th.statement_text,
                        "proof": th.proof_text,
                        "symbols": [asdict(su) for su in th.symbols],
                        "term": th.term
                    })
                
                with open(export_path, 'w') as file:
                    json.dump({filename: data}, file, indent=4)
            except Exception as e:
                print(e)

    extractor.close()


if __name__ == "__main__":
    main()
