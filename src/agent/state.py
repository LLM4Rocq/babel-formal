from enum import StrEnum
from typing import List
import re

from transformers import AutoTokenizer

class BlockType(StrEnum):
    INSTRUCTION = "instruction"
    THINK = "think"
    SCRIPT = "script"

class Block:
    def __init__(
        self,
        kind: BlockType,
        text: str,
        to_continue=False
    ):
        self.kind = kind
        self.text = text
        self.to_continue=to_continue

    def dump(self):
        """
        Render block according to kind and status
        """
        if self.kind == BlockType.THINK:
            str_kind = str(self.kind)
            if self.to_continue:
                return f"\n<{str_kind}>{self.text}"
            else:
                return f"\n<{str_kind}>{self.text}</{str_kind}>"
        elif self.kind == BlockType.SCRIPT:
            return f"\n\\box{{{self.text}}}"
        return self.text

    def resume(self, text: str="", to_continue=False):
        """
        Close to_continue block status
        """
        self.text += text
        self.to_continue = to_continue
    
    def __str__(self):
        return f"Kind:{str(self.kind)}\nTo continue: {self.to_continue}\nText:\n{self.text}"

class StateError(Exception):
    pass

class State:
    def __init__(self, instruction: str):
        self.blocks = [Block(BlockType.INSTRUCTION, instruction)]
    
    def _parse_output(self, output: str) -> List[Block]:
        """
        Extracts:
        - <think>...</think> blocks
        - \\box{...} script blocks
        - implicit think blocks (text outside tags, ending at </think>)
        """
        token_pattern = re.compile(
            r"(<think>.*?</think>)"        # full think block
            r"|"
            r"(\\box\{.*?\})"              # script block
            r"|"
            r"(</think>)",                 # implicit think terminator
            re.DOTALL
        )
        THINK_OPEN = "<think>"
        THINK_CLOSE = "</think>"
        blocks: List[Block] = []
        pos = 0                          # current scan position
        pending_think = ""               # buffer for implicit THINK

        for m in token_pattern.finditer(output):
            start, end = m.span()
            think_tag, script_tag, implicit_close = m.groups()

            # Text between tokens
            preceding = output[pos:start]

            if preceding.strip():
                # Pre-token text belongs to implicit THINK
                pending_think += preceding

            if think_tag:
                # Flush any pending implicit think
                if pending_think.strip():
                    blocks.append(Block(BlockType.THINK, pending_think.strip()))
                    pending_think = ""

                # Extract explicit think content
                content = think_tag[len(THINK_OPEN):-len(THINK_CLOSE)].strip()
                blocks.append(Block(BlockType.THINK, content))

            elif script_tag:
                # Flush pending implicit think
                if pending_think.strip():
                    blocks.append(Block(BlockType.THINK, pending_think.strip()))
                    pending_think = ""

                # Extract script content
                content = script_tag[len(r"\box{"):-1].strip()
                blocks.append(Block(BlockType.SCRIPT, content))

            elif implicit_close:
                # Implicit closing of a THINK block
                if pending_think.strip():
                    blocks.append(Block(BlockType.THINK, pending_think.strip()))
                pending_think = ""

            pos = end

        # Trailing text after last token:
        tail = output[pos:]
        if tail.strip():
            pending_think += tail
            blocks.append(Block(BlockType.THINK, pending_think.strip()))

        return blocks

    def parse_output(self, output: str) -> List[Block]:
        """
        Pattern extracts: <think>...</think>, \\box{...}, and any plain text
        """
        blocks = self._parse_output(output)
        if len(blocks) != 2 or blocks[-1].kind != BlockType.SCRIPT:
            raise StateError("Output doesn't satisfy the expected output format.")
        return blocks

    def update(self, new_blocks: List[Block]):
        """
        Update old block/Append new blocks
        """
        if self.blocks and self.blocks[-1].to_continue:
            if self.blocks[-1].kind != new_blocks[0].kind:
                raise StateError("First generated block does not continue last block as required.")
            self.blocks[-1].resume(new_blocks[0].text, to_continue=False)
            new_blocks = new_blocks[1:]
        self.blocks += new_blocks

    def amend_last_block(self, text: str, to_continue: bool=True):
        """
        Append text to last block, by default, change last state into a partial block that need to be completed.
        """
        self.blocks[-1].resume(text, to_continue=to_continue)

    def add_block(self, kind: BlockType, text: str, to_continue=False):
        """
        Add a block to the current state
        """
        self.blocks += [Block(kind, text, to_continue=to_continue)]

    def rollback_before(self, block: Block):
        """
        Remove everything after the given block (included).
        """

        for i in range(len(self.blocks) - 1, -1, -1):
            if self.blocks[i] == block:
                self.blocks = self.blocks[: i]
                return
        raise StateError("Block not Found")

    def dump_messages(self):
        """
        Fold blocks into role-based message list
        """
        messages = []
        prev_role = ""
        for block in self.blocks:
            role = "user" if block.kind == BlockType.INSTRUCTION else "assistant"

            if role != prev_role:
                messages.append({"role": role, "content": block.dump()})
            else:
                messages[-1]["content"] += block.dump()

            prev_role = role
        return messages

    def dump_prompt(self, tokenizer: AutoTokenizer):
        """
        Fold blocks into a prompt
        """
        messages = self.dump_messages()
        if not messages:
            raise StateError('No blocks found in current state')
        if messages[-1]['role'] == 'assistant':
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_text += messages[-1]['content']
        else:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        return prompt_text

    def __str__(self):
        output = ""
        for k, block in enumerate(self.blocks, 1):
            output += f"{k}th Block:\n" + str(block) + "\n\n"
        return output[:-2]
