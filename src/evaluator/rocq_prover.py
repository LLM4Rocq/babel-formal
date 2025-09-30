from __future__ import annotations

import os
import shutil
import random
import time
import subprocess
from typing import Optional, List

from pytanque import Pytanque, PetanqueError

from .prover import Prover, DatasetItem, Message, MessageType

class RocqProver(Prover):
    """
    Coq/Rocq implementation over pytanque + pet-server.

    Notes:
      - If start_server=True, we spawn a local pet-server on the chosen port.
      - We copy the .v source into an aux file (…aux_ssreflect.v) to avoid
        touching your originals, following your script.
    """
    def __init__(
        self,
        dataset_dir: str,
        url: str = "127.0.0.1",
        port: int = 8765,
        timeout_per_step: int = 10,
        start_server: bool = False,
        mean_wait: int = 10,
    ):
        super().__init__(dataset_dir)
        self.url = url
        self.port = port
        self.timeout_per_step = timeout_per_step
        self._pet = None
        self._pet_proc: Optional[subprocess.Popen] = None
        self._aux_file: Optional[str] = None
        self._state = None
        self._start_server = start_server
        self._mean_wait = mean_wait

    # --- server lifecycle
    def _spawn_server_if_needed(self):
        if not self._start_server or self._pet_proc is not None:
            return
        self._pet_proc = subprocess.Popen(
            ["pet-server", "--port", str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # give it a short randomized warm-up like your script
        time.sleep(random.randint(1, 2 * max(1, self._mean_wait)))

    def _connect(self):
        self._pet = Pytanque(self.url, self.port)
        self._pet.connect()

    def _disconnect(self):
        if self._pet is not None:
            self._pet.close()
            self._pet = None

    def close(self):
        self._disconnect()
        if self._pet_proc is not None:
            self._pet_proc.terminate()
            try:
                self._pet_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._pet_proc.kill()
                self._pet_proc.wait()
            self._pet_proc = None

    # --- Prover API

    def start_thm(self, item: DatasetItem) -> List[str]:
        assert item.prover == "rocq", "RocqProver can only handle 'rocq' items."
        if not item.name:
            raise ValueError("Rocq item missing theorem/lemma 'name'.")

        self._spawn_server_if_needed()
        if self._pet is None:
            self._connect()

        # Prepare an aux copy of the file
        orig_v = os.path.join(self.dataset_dir, f"{item.source}.v")
        if not os.path.exists(orig_v):
            raise FileNotFoundError(f"Coq file not found: {orig_v}")
        aux_v = os.path.join(self.dataset_dir, f"{item.source}_aux_ssreflect.v")
        shutil.copyfile(orig_v, aux_v)
        self._aux_file = aux_v
        self._current_item = item
        # Start the proof
        try:
            self._pet.set_workspace(True, "")
            self._state = self._pet.start(file=self._aux_file, thm=item.name)
        except PetanqueError as e:
            self._state = None
            return []

        goals = [g.pp for g in self._pet.goals(self._state)]
        return goals

    def check_proof(self, proof: str) -> Message:
        proof = [step + '.' for step in proof.split('.') if step]
        new_proof = []
        for step in proof:
            step = step.strip()
            found = False
            for sym in ['-', '+', '*']:
                if step.startswith(sym):
                    new_proof.append(sym)
                    new_proof.append(step[1:])
                    found = True
                    break
            if not found:
                new_proof.append(step)
        for step in new_proof:
            message = self.run_tac(step)
            if message.status == MessageType.ERROR:
                return message
        return message
        

    def run_tac(self, tactic: str) -> Message:
        if self._pet is None or self._state is None or self._current_item is None:
            raise RuntimeError("start_thm must be called before run_tac().")

        try:
            self._state = self._pet.run(self._state, tactic, verbose=False, timeout=self.timeout_per_step)
            goals = [g.pp for g in self._pet.goals(self._state)]
            status = MessageType.FINISH if getattr(self._state, "proof_finished", False) else MessageType.ONGOING
            return Message(status=status, goals=goals, tactic=tactic)
        except PetanqueError as e:
            return Message(status=MessageType.ERROR, goals=[], tactic=tactic, message=e.message)
