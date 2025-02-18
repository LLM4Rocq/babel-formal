from typing import Optional, List

import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'coqpyt/'))

from coqpyt.coq.structs import TermType, Step, ProofStep, ProofTerm
from coqpyt.coq.proof_file import ProofFile



class ProofFileLight(ProofFile):
    """"Lighter version of ProofFile (no recursive Locate search during context retrieval), to avoid recursive blow-up 
    """

    def __step(self, step: Step, undo: bool):
        aux_file = self._ProofFile__aux_file
        file_change = aux_file.truncate if undo else aux_file.append
        file_change(step.text)
        # Ignore segment delimiters because it affects Program handling
        if self.context.is_segment_delimiter(step):
            return
        # Found [Qed]/[Defined]/[Admitted] or [Proof <exact>.]
        if self.context.is_end_proof(step):
            self.__handle_end_proof(step, undo=undo)
        # New obligations were introduced
        elif self._ProofFile__has_obligations(step):
            self.__handle_obligations(step, undo=undo)
        # Check if proof step
        elif len(self.open_proofs) > 0 if undo else self.in_proof:
            self.__check_proof_step(step, undo=undo)

    def exec(self, nsteps=1) -> List[Step]:
        sign = 1 if nsteps > 0 else -1
        initial_steps_taken = self.steps_taken
        nsteps = min(
            nsteps * sign,
            len(self.steps) - self.steps_taken if sign > 0 else self.steps_taken,
        )
        step = lambda: self.prev_step if sign == 1 else self.curr_step

        for _ in range(nsteps):
            # HACK: We ignore steps inside a Module Type since they can't
            # be used outside and should be overriden.
            in_module_type = self.context.in_module_type
            self._step(sign)
            if in_module_type or self.context.in_module_type:
                continue
            self.__step(step(), sign == -1)

        last, slice = sign == 1, (initial_steps_taken, self.steps_taken)
        return self.steps[slice[1 - last] : slice[last]]

    def __handle_end_proof(
        self,
        step: Step,
        index: Optional[int] = None,
        open_index: Optional[int] = None,
        undo: bool = False,
    ):
        proofs = self._ProofFile__proofs
        open_proofs = self._ProofFile__open_proofs
        goals = self._ProofFile__goals
        if undo:
            index = -1 if index is None else index
            open_index = len(open_proofs) if open_index is None else open_index
            proof = proofs.pop(index)
            proof.steps.pop()
            open_proofs.insert(open_index, proof)
        else:
            index = len(self._ProofFile__proofs) if index is None else index
            open_index = -1 if open_index is None else open_index
            open_proof = open_proofs.pop(open_index)
            # The goals will be loaded if used (Lazy Loading)
            open_proof.steps.append(ProofStep(step, goals, []))
            proofs.insert(index, open_proof)
    
    def __check_proof_step(self, step: Step, undo: bool = False):
        # Avoids Tactics, Notations, Inductive...
        if self.context.term_type(step) == TermType.OTHER:
            self.__handle_proof_step(step, undo=undo)
        elif self.context.is_proof_term(step):
            self.__handle_proof_term(step, undo=undo)

    def __handle_proof_step(self, step: Step, undo: bool = False):
        open_proofs = self._ProofFile__open_proofs
        goals = self._ProofFile__goals
        if undo:
            open_proofs[-1].steps.pop()
        else:
            # The goals will be loaded if used (Lazy Loading)
            open_proofs[-1].steps.append(ProofStep(step, goals, ""))

    def __handle_proof_term(
        self, step: Step, index: Optional[int] = None, undo: bool = False
    ):
        open_proofs = self._ProofFile__open_proofs
        if undo:
            index = -1 if index is None else index
            open_proofs.pop(index)
        else:
            index = len(open_proofs) if index is None else index
            # New proof terms can be either obligations or regular proofs
            proof_term = self.context.last_term
            open_proofs.insert(
                index, ProofTerm(proof_term, "", [], None)
            )