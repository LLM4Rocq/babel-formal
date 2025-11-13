from abc import ABC, abstractmethod
from enum import StrEnum
from typing import List

from src.evaluator.prover import ProverError, Prover, Message, MessageType, DatasetItem
from src.llm.base import BaseLLM
from src.agent.state import State, BlockType
from src.agent.base import BaseAgent, AgentStatus


class RocqAgent(BaseAgent):
    """Agent class for Rocq prover"""

    def start_thm(self, item: DatasetItem):
        initial_goals = "\n".join(self.prover.start_thm(item))
        instruction = self._instruction(item)
        self.state = State(instruction)
        start_think = f" Okay, let\'s try to transform this proof term into a sequence of coq tactics. First let\'s write down the hypotheses, and the initial goal (after the \"|-\" symbol) given by the coq proof assistant:\n{initial_goals}."
        self.state.add_block(BlockType.THINK, start_think, to_continue=True)
    
    def _instruction(self, item: DatasetItem):
        instr = f"You are given a proof term:\n\n{item.term}\n\nYour task is to derive a sequence of tactics that corresponds to this term.\n\nWhen you work through the problem, write down your reasoning in detail inside <think> ... </think> tags. This reasoning should reflect your natural thought process as you explore the structure of the term and figure out what tactics to apply. You should consider different possible approaches, reflect on why some might or might not work, and gradually converge on a tactic choice.\n\nAfter each reasoning block, provide the next (group of) tactic(s) enclosed in:\n\n\\box{{\n  <tactic>\n}}\n\nSome dependencies that could be helpful:\n\n{item.dependencies}"
        return instr
    