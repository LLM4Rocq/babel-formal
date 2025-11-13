from __future__ import annotations
import os
import json
from typing import Optional, List, Dict, Any

import leanclient as lc

from .prover import Prover, DatasetItem, Message, MessageType, ProverError

class LeanProver(Prover):
    """
    Lean implementation over Lean LSP (leanclient).

    Approach:
      - We open the file via LeanLSPClient.create_file_client.
      - We track the proof region [line_start, line_end).
      - Each run_tac appends the tactic line to the region text and updates the document.
      - We use diagnostics to detect errors; goals are retrieved via get_goal(line_start, 0).
    """
    def __init__(self, dataset_dir: str):
        super().__init__(dataset_dir)
        self._client = lc.LeanLSPClient(dataset_dir)
        self._sfc = None
        self._prev_buffer: Optional[str] = None
        self._line_start: Optional[int] = None
        self._line_end: Optional[int] = None
        self._current_item: Optional[DatasetItem] = None
        self._accumulated: str = ""  # accumulated proof text including trailing newline

    @classmethod
    def name(cls):
        return "Lean 4 prover"

    def _check_error(self, diags: List[Dict[str, Any]]) -> Optional[str]:
        """
        Return error message if any diagnostic has severity == 1 (error),
        otherwise None.
        """
        for d in diags or []:
            if d.get("severity") == 1:
                if d.get('message').startswith('unsolved goals'):
                    continue
                # build a compact message
                msg = d.get("message", "Lean error")
                rng = d.get("range") or {}
                return f"{msg} @ {rng}"
        return None

    def _get_goals(self, diags: List[Dict[str, Any]]) -> List[str]:
        goals = []
        for d in diags or []:
            if d.get("severity") == 1:
                if d.get('message').startswith('unsolved goals'):
                    goals.append(d.get('message'))
        return goals

    def _reset_current(self):
        if self._current_item:
            change = lc.DocumentContentChange(
                text=self._buffer,
                start=[self._line_start, 0],
                end=[self._line_end, 0],
            )
            self._sfc.update_file(changes=[change])
            self._current_item = None

    def start_thm(self, item: DatasetItem) -> List[str]:
        self._reset_current()
        self._current_item = item
        assert item.prover == "lean", "LeanProver can only handle 'lean' items."
        if item.lines is None:
            raise ValueError("Lean item missing 'lines' range.")
        
        with open(os.path.join(self.dataset_dir, item.source + '.lean')) as file:
            content = file.read()
        self._current_item = item
        # Open file client
        lean_path = f"{item.source}.lean"
        self._sfc = self._client.create_file_client(lean_path)
        self._line_start, self._line_end = item.lines
        self._buffer = "\n".join(content.splitlines()[item.lines[0]: item.lines[1]+1])
        self._accumulated = ""
        self._sfc.get_diagnostics()
        change = lc.DocumentContentChange(
            text=self._accumulated,
            start=[self._line_start, 0],
            end=[self._line_end, 0],
        )
        self._line_end = self._line_start
        self._sfc.update_file(changes=[change])
        diags = self._sfc.get_diagnostics()
        return self._get_goals(diags)

    def check_proof(self, proof: str) -> Message:
        return self.run_tac(proof)
    
    def run_tac(self, tactic: str) -> Message:
        tactic = tactic.rstrip()
        if self._sfc is None or self._line_start is None or self._line_end is None:
            raise RuntimeError("start_thm must be called before run_tac().")
        self._accumulated = (self._accumulated + f"{tactic}\n")
        change = lc.DocumentContentChange(
            text=self._accumulated,
            start=[self._line_start, 0],
            end=[self._line_end, 0],
        )
        self._sfc.update_file(changes=[change])
        self._line_end = self._line_start + self._accumulated.count('\n')

        diags = self._sfc.get_diagnostics()
        err = self._check_error(diags)    
        goals = self._get_goals(diags)
        if err or goals:
            return Message(status=MessageType.ONGOING, goals=goals, tactic=tactic, message=err)
        return Message(status=MessageType.FINISH, goals=[], tactic=tactic)

    def close_proof(self):
        return
