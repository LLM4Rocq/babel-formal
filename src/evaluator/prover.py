from __future__ import annotations
from abc import ABC, abstractmethod


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
    term: Optional[str] = ""
    dependencies: Optional[str] = ""

    # For Rocq:
    name: Optional[str] = None  # theorem/lemma name (Rocq)
    proof_text: Optional[str] = None
    # For Lean:
    lines: Optional[Tuple[int, int]] = None  # (start_line, end_line) for Lean

class ProverError(Exception):
    def __init__(self, message, prover_feedback=""):
        super().__init__(message)
        self.prover_feedback = prover_feedback

class Prover(ABC):
    """
    Abstract prover class 
    """
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self._current_item: Optional[DatasetItem] = None

    @classmethod
    @abstractmethod
    def name(self) -> str:
        "Return prover name"
        pass

    @abstractmethod
    def start_thm(self, item: DatasetItem) -> List[str]:
        """Prepare the theorem/proof state and return initial goals."""
        pass

    @abstractmethod
    def run_tac(self, tactic: str) -> Message:
        """Run a single tactic step and return current goals + status."""
        pass
    
    @abstractmethod
    def check_proof(self, proof: str) -> Message:
        """Run a whole proof and return current goals + status."""
        pass

    @abstractmethod
    def close_proof(self):
        """Final check to make sure current proof is complete."""
        pass