from .prover import Prover
from .rocq_prover import RocqProver
from .lean_prover import LeanProver

def make_prover(kind: str, dataset_dir: str, **kwargs) -> Prover:
    kind = kind.lower()
    if kind == "rocq":
        return RocqProver(dataset_dir, **kwargs)
    if kind == "lean":
        return LeanProver(dataset_dir)
    raise ValueError(f"Unknown prover kind: {kind}")