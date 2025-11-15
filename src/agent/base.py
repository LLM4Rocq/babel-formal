from abc import ABC, abstractmethod
from enum import StrEnum

from src.evaluator.prover import Prover, MessageType, DatasetItem
from src.llm.base import BaseLLM
from src.agent.state import State, BlockType


class AgentStatus(StrEnum):
    ONGOING = "ongoing"
    FINISH = "finish"
    FATAL = "fatal"

class BaseAgent(ABC):
    """Abstract base class for Agent."""

    def __init__(
        self,
        llm: BaseLLM,
        prover: Prover,
        max_retry: int=3,
        max_depth: int=10,
        feedback_error: bool=False,
        feedback_goals: bool=False
    ):
        self.llm = llm
        self.prover = prover
        self.state: State = None
        self.status = AgentStatus.ONGOING
        self.max_retry = max_retry
        self.max_depth = max_depth

        self.feedback_error = feedback_error
        self.feedback_goals = feedback_goals
        self.current_depth = 0
        self.num_errors = 0
        self.logs = []

    @abstractmethod
    def start_thm(self, item:DatasetItem):
        pass
    
    def step(self):
        """Apply one step"""
        if self.status == AgentStatus.FINISH or self.status == self.status.FATAL:
            return
        if self.num_errors >= self.max_retry or self.current_depth >= self.max_depth:
            self.status = AgentStatus.FATAL
            return 
        prompt = self.state.dump_prompt(self.llm.tokenizer)
        output = self.llm.generate(prompt)
        self.logs.append({"prompt": prompt, "output": output})
        new_blocks = self.state.parse_output(output)
        message = None
        try:
            message = self.prover.run_tac(new_blocks[-1].text)
            if message.status == MessageType.FINISH:
                self.prover.close_proof()
                self.status = AgentStatus.FINISH
                return
            if message.status == MessageType.ERROR:
                self.num_errors += 1
                if self.feedback_error:
                    script = new_blocks[-1].text
                    amend = f"Wait, when I wrote \\box{{{script}}}, I received this feedback from {self.prover.name()}: {message.message[:100]}."
                    self.state.amend_last_block(amend, to_continue=True)
                    self.num_errors += 1
                return
            self.state.update(new_blocks)
            self.current_depth += 1
            self.num_errors = 0
        except Exception as e:
            out_state = str(self.state)
            self.logs.append({
                "in_state": str(self.state),
                "out_state": out_state,
                "output": output,
                "error": str(e)
            })
            self.num_errors += 1
        if self.feedback_goals and message:
            new_goals = "\n".join(message.goals)
            positive_feedback = f"Let's continue to translate this proof term into a proof script. {self.prover.name()} gives me these new goals: {new_goals}."
            self.state.add_block(BlockType.THINK, positive_feedback, to_continue=True)
    
    def try_proof(self, item: DatasetItem):
        self.start_thm(item)
        while self.status != AgentStatus.FATAL and self.status != AgentStatus.FINISH:
            self.step()

    
            
