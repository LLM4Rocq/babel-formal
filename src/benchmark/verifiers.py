from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.evaluator.prover import DatasetItem, MessageType


@dataclass
class VerificationResult:
    status: MessageType
    goals: List[str]
    error: str
    raw: Dict[str, object]


class BaseVerifier:
    def verify(self, example, proof: str) -> VerificationResult:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        return None


class RocqLocalVerifier(BaseVerifier):
    def __init__(
        self,
        workspace: str,
        host: str,
        port: int,
        timeout_per_step: int,
        start_server: bool,
    ):
        from src.evaluator.rocq_prover import RocqProver

        self.workspace = workspace
        self._prover = RocqProver(
            workspace,
            url=host,
            port=port,
            timeout_per_step=timeout_per_step,
            start_server=start_server,
        )

    def verify(self, example, proof: str) -> VerificationResult:
        item = DatasetItem(
            prover="rocq",
            source=example.source_stem,
            name=example.theorem_name,
            lines=(0, 0),
        )
        try:
            self._prover.start_thm(item)
            message = self._prover.check_proof(proof)
        except Exception as exc:  # pragma: no cover - runtime errors from prover backend
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error=str(exc),
                raw={},
            )

        return VerificationResult(
            status=message.status,
            goals=list(message.goals),
            error=str(message.message or ""),
            raw={},
        )

    def close(self) -> None:
        close_fn = getattr(self._prover, "close", None)
        if callable(close_fn):
            close_fn()


class RocqMLServerVerifier(BaseVerifier):
    def __init__(
        self,
        workspace: str,
        host: str,
        port: int,
        timeout_per_step: int,
    ):
        from rocq_ml_toolbox.inference.client import PytanqueExtended

        self.workspace = workspace
        self.timeout_per_step = timeout_per_step
        self._client = PytanqueExtended(host, port)
        self._client.connect()

    @staticmethod
    def _split_proof_steps(proof: str) -> List[str]:
        raw_steps = [chunk.strip() for chunk in proof.split(".") if chunk.strip()]
        steps: List[str] = []
        for step in raw_steps:
            matched = False
            for symbol in ("-", "+", "*"):
                if step.startswith(symbol):
                    steps.append(symbol)
                    rest = step[1:].strip()
                    if rest:
                        steps.append(rest)
                    matched = True
                    break
            if not matched:
                steps.append(step)
        return steps

    def verify(self, example, proof: str) -> VerificationResult:
        source_file = Path(self.workspace) / f"{example.source_stem}.v"
        if not source_file.exists():
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error=f"Rocq source file not found: {source_file}",
                raw={},
            )

        try:
            self._client.set_workspace(True, "")
            state = self._client.start(file=str(source_file), thm=example.theorem_name)
        except Exception as exc:
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error=str(exc),
                raw={},
            )

        try:
            message = ""
            for step in self._split_proof_steps(proof):
                state = self._client.run(state, step + ".", verbose=False, timeout=self.timeout_per_step)
            goals = [g.pp for g in self._client.goals(state)]
            finished = bool(getattr(state, "proof_finished", False))
            return VerificationResult(
                status=MessageType.FINISH if finished else MessageType.ONGOING,
                goals=goals,
                error=message,
                raw={},
            )
        except Exception as exc:
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error=str(exc),
                raw={},
            )

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()


class LeanKiminaVerifier(BaseVerifier):
    def __init__(
        self,
        workspace: str,
        url: str,
        timeout_seconds: int,
        api_key: Optional[str],
    ):
        self.workspace = workspace
        self.url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _build_code(self, example, proof: str) -> str:
        if example.target_lines is None:
            raise ValueError("Lean example is missing target lines.")

        source_file = Path(self.workspace) / f"{example.source_stem}.lean"
        if not source_file.exists():
            raise FileNotFoundError(f"Lean source file not found: {source_file}")

        start, end = example.target_lines
        lines = source_file.read_text(encoding="utf-8").splitlines()
        before = lines[:start]
        after = lines[end + 1 :]
        proof_lines = proof.splitlines() if proof else []
        merged = before + proof_lines + after
        return "\n".join(merged)

    @staticmethod
    def _to_result(payload: Dict[str, object]) -> VerificationResult:
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error="Invalid Kimina response: missing results.",
                raw=payload,
            )

        result = results[0]
        if not isinstance(result, dict):
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error="Invalid Kimina response item.",
                raw=payload,
            )

        if result.get("error"):
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error=str(result.get("error")),
                raw=result,
            )

        response = result.get("response")
        if not isinstance(response, dict):
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error="Invalid Kimina response: missing response object.",
                raw=result,
            )

        # REPL-level errors come with a single `message` field.
        if "message" in response:
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error=str(response.get("message")),
                raw=result,
            )

        messages = response.get("messages")
        errors: List[str] = []
        goals: List[str] = []
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                severity = message.get("severity")
                data = str(message.get("data", ""))
                if severity == "error":
                    errors.append(data)
                elif data and ("goal" in data.lower() or "⊢" in data):
                    goals.append(data)

        sorries = response.get("sorries")
        if isinstance(sorries, list) and sorries:
            for entry in sorries:
                if isinstance(entry, dict):
                    goal = entry.get("goal")
                    if goal:
                        goals.append(str(goal))
            errors.append("Proof contains `sorry`.")

        if errors:
            return VerificationResult(
                status=MessageType.ERROR,
                goals=goals,
                error="\n".join(errors),
                raw=result,
            )

        return VerificationResult(
            status=MessageType.FINISH,
            goals=goals,
            error="",
            raw=result,
        )

    def verify(self, example, proof: str) -> VerificationResult:
        try:
            merged_code = self._build_code(example, proof)
        except Exception as exc:
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error=str(exc),
                raw={},
            )

        body = {
            "codes": [
                {
                    "custom_id": example.example_id,
                    "proof": merged_code,
                }
            ],
            "timeout": int(self.timeout_seconds),
            "disable_cache": False,
        }

        try:
            response = self.session.post(
                f"{self.url}/verify",
                data=json.dumps(body),
                headers=self.headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error=str(exc),
                raw={},
            )

        if not isinstance(payload, dict):
            return VerificationResult(
                status=MessageType.ERROR,
                goals=[],
                error="Invalid Kimina response payload.",
                raw={},
            )

        return self._to_result(payload)


def build_lean_verifier(
    workspace: str,
    kimina_url: str,
    kimina_timeout: int,
    kimina_api_key: Optional[str],
) -> BaseVerifier:
    return LeanKiminaVerifier(
        workspace=workspace,
        url=kimina_url,
        timeout_seconds=kimina_timeout,
        api_key=kimina_api_key,
    )


def build_rocq_verifier(
    backend: str,
    workspace: str,
    host: str,
    port: int,
    timeout_per_step: int,
    start_server: bool,
) -> BaseVerifier:
    if backend == "ml_server":
        return RocqMLServerVerifier(
            workspace=workspace,
            host=host,
            port=port,
            timeout_per_step=timeout_per_step,
        )
    return RocqLocalVerifier(
        workspace=workspace,
        host=host,
        port=port,
        timeout_per_step=timeout_per_step,
        start_server=start_server,
    )
