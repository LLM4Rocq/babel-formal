# Babel-Formal notes

This document is a hands-on complement to the paper. It walks through the two translation tracks (Lean↔Rocq and vanilla Rocq→SSReflect), the datasets we build, and the scripts that orchestrate each step.

## 1. Translation tasks
- **Cross-assistant.** Given aligned statements, translate Lean scripts into Rocq and Rocq scripts into Lean by leaning on proof terms.
- **Cross-tactic.** Convert vanilla Rocq scripts into SSReflect style inside MathComp.
- Both directions share the same idea: pretty-print the proof term, let a model reason about it, and regenerate a native script in the target dialect.

## 2. Data snapshot
- **Aligned benchmark.** 14 paired files, 117 lemmas. Each lemma has a Lean version (`dataset/*.lean`) and a Rocq version (`dataset/repo/rocq/*.v`).
- **Training pools.**
  - Lean: 500 Mathlib proofs after filtering.
  - Rocq (cross-assistant): 500 C-CoRN proofs.
  - Rocq (SSReflect): 1000 MathComp entries.
- **Proof dumps.** `src/proof_dumps/` is the fastest way to recover proof terms and apply quick heuristics (length, token budget) before using the full extraction pipeline.
- **Reasoning traces.** Backward chains produced with Gemini 2.5 Pro (`export/dataset/step_7`, `export_dataset_rocq`).

## 3. Lean data pipeline
If you only need proof-term statistics, stop at `src/proof_dumps/` and filter there; the remaining steps assemble the full training-ready dataset.

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `src/lean_rocq_translation/step_1/exec.py` | Traverse Mathlib with `leanclient`, recover statements, proof bodies, and LSP ranges. |
| 2 | `step_2/exec.py` | Align extracted entries with the proof dump and merge statement + term (`lean_extract`). |
| 3 | `step_3/exec.py` | Filter by term length / proof length, then select a diverse subset using BM25 round-robin. |
| 4 | `step_4/exec.py` | Resolve fully-qualified names, store symbol metadata, and capture canonical proof terms (`#print`). |
| 5 | `step_5/exec.py` | Re-run each proof with `lean_evaluate` to make sure the snippet is self-contained and still checks. |
| 6 | `step_6/exec.py` | Sampling by proof length; keeps symbol annotations only. |
| 7 | `step_7/exec.py` | Generate backward reasoning traces with Gemini 2.5 Pro using `prompt.txt` and `config.yaml`. |

Each directory keeps the JSON export of the step so the pipeline is restartable. Use the `--output` flag to redirect intermediate files if you need a custom run.

## 4. Rocq data pipeline
- **Extraction (`export_dataset_lean_rocq/merge.py`, `src/inference/generate_*`).** Collect Rocq statements, proof terms, and tactic scripts from MathComp or C-CoRN.
- **Reasoning traces (`src/lean_rocq_translation/step_7_rocq/exec.py`).** Prompt Gemini with Rocq proof terms, scripts, constants, and notations.
- **Packaging (`misc/step_8/exec.py`).** Same as Lean: convert traces into the training JSON layout.

All Rocq exports land in `export_dataset_rocq/` by default.

## 5. Training the Babel models
- **Models.** `Qwen2.5-Coder-32B-Instruct` fine-tuned with two configurations:
  - `Babel-translate`: map Lean terms to Lean scripts and Rocq terms to Rocq scripts.
  - `Babel-ssreflect`: map Rocq terms (vanilla) to SSReflect scripts.
- **Objective.** Only compute loss on reasoning spans and the final `<tactic>` blocks.
- **Schedule.** 5 epochs, batch size 16, AdamW (β₁ = 0.9, β₂ = 0.95, weight decay 1e‑4), learning rate 1e‑5 with 5 % warmup and cosine decay.
- **Implementation.** Training scripts live in `src/training_nemo/` and build on NVIDIA NeMo; adjust hyperparameters in `config/training/nemo.yaml` and launch runs with the SLURM jobs under `config/training/training_h100.slurm` (train) and `config/training/eval_h100.slurm` (eval).
- **Hardware.** 8×4 NVIDIA H100, ~1 hour per run (bfloat16).
- **Prompts.** Stored under `config/prompts` and mirrored for Rocq/SSReflect.

## 6. Inference pipelines
1. **Prepare context.** Extract the source proof term plus dependencies (`lean_extract['term']`, Rocq constants/notations).
2. **Sample candidates.** Use the fine-tuned model to generate up to 128 sequences, each interleaving `<think>` reasoning and `<tactic>` blocks.
3. **Optional fusion.** Call GPT‑5 with script-to-script translation (see `src/evaluation/model.py` and `misc/evaluation/*`). Union the candidates if you want the paper’s combined numbers.
4. **Check scripts.** Run `src/evaluation/lean_evaluate.py` or `src/evaluation/rocq_evaluate.py` to replay tactics and compute pass@k.

`src/inference/generate_base_to_rocq.py` and friends show how we bind all the steps for batched translation.

## 7. Evaluation recipe
- **Config.** Choose a benchmark in `src/evaluation/config.yaml` (Lean↔Rocq or SSReflect track).
- **Run.** `python src/evaluation/rocq_evaluate.py --config src/evaluation/config.yaml` (or the Lean variant). The scripts integrate with Pytanque for Rocq and `leanclient` for Lean.
- **Metrics.** pass@k (non-interactive) for Babel models, pass@1 with feedback loops for GPT‑5. Combined scores take the union of both candidate sets.

## 9. Tips & troubleshooting
- Long proof terms: revisit `step_3` filters or adjust the tokenizer limits before fine-tuning.
- Missing dependencies: regenerate symbol metadata (`step_4` for Lean, Rocq extractors for Coq) so the prompts always list the needed lemmas.
