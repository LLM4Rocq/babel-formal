from __future__ import annotations

import os
import json

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, Optional, Tuple, List

class MessageType(Enum):
    ONGOING = auto()
    ERROR = auto()
    FINISH = auto()

@dataclass
class Message:
    status: MessageType
    goals: List[str]
    tactic: str
    message: str = ""   # error text or extra info

@dataclass
class DatasetItem:
    """A normalized dataset item covering both Lean and Rocq entries."""
    prover: str                 # "lean" | "rocq"
    source: str                 # source file stem (without extension)
    # For Rocq:
    name: Optional[str] = None  # theorem/lemma name (Rocq)
    proof_text: Optional[str] = None
    # For Lean:
    lines: Optional[Tuple[int, int]] = None  # (start_line, end_line) for Lean

# ------- Abstract Prover

class Prover:
    """
    Unifies the interaction with provers.

    Lifecycle:
      - create instance with dataset_dir
      - iterate items via .iter_items()
      - call start_thm(item) -> initial goals
      - repeatedly call run_tac("...") -> Message (status + current goals)
    """
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self._current_item: Optional[DatasetItem] = None

    # ---- API surface you asked for
    def start_thm(self, item: DatasetItem) -> List[str]:
        """Prepare the theorem/proof state and return initial goals."""
        raise NotImplementedError

    def run_tac(self, tactic: str) -> Message:
        """Run a single tactic step and return current goals + status."""
        raise NotImplementedError

    def check_proof(self, proof: str) -> Message:
        """Run a single tactic step and return current goals + status."""
        raise NotImplementedError
    
    # ---- Dataset loading (shared)
    def iter_items(self) -> Iterator[DatasetItem]:
        """
        Iterate all dataset items in dataset/json/*.json and normalize them
        into DatasetItem objects for the concrete prover.
        """
        json_dir = os.path.join(self.dataset_dir, "json")
        for fname in sorted(os.listdir(json_dir)):
            with open(os.path.join(json_dir, fname), "r") as fh:
                content = json.load(fh)
            source = content["source"]

            for item in content["items"]:
                # Lean path
                if "lean" in item:
                    le = item["lean"]
                    yield DatasetItem(
                        prover="lean",
                        source=source,
                        proof_text=le.get("proof", ""),
                        lines=tuple(le["lines"]),
                        name=None,
                    )
                # Rocq path
                if "coq" in item:
                    rq = item["coq"]
                    yield DatasetItem(
                        prover="rocq",
                        source=source,
                        name=item.get("name"),
                        proof_text=rq.get("proof", ""),
                        lines=None,
                    )

