# Babel-Formal
This repository is anonymized for review. Trained models will be released if the paper is accepted.

Babel-Formal explores proof term translation as a practical bridge across interactive theorem provers. We translate proofs between Lean and Rocq and across tactic sets by treating proof terms as a pivot language.

## Project scope
- Translate Lean scripts into Rocq and Rocq scripts into Lean using aligned proof terms instead of parallel scripts.
- Transfer Rocq proofs written with vanilla tactics into SSReflect style, again using proof terms as the interface.
- Release the aligned benchmark (14 files, 117 lemmas), training traces, reasoning prompts.
- Release two models, a first one to translate between Lean and Rocq, and a second one to go from vanilla Rocq to SSReflect.

## Approach at a glance
- **Proof subset selection.** Sample diverse Mathlib, C-CoRN, and MathComp proofs (1k entries per setup) after filtering by length.
- **Proof term extraction.** Use `coqpyt` to recover statements, local context, terms, and notation blocks.
- **Proof dumps for prefiltering.** `src/proof_dumps/` provides the fastest way to recover proof terms and apply quick length/token filters before running the full pipeline.
- **Backward reasoning traces.** Ask Gemini 2.5 Pro to simulate step-by-step reasoning that recreates each target script from its proof term.
- **Fine-tuning.** Train two variants of `Qwen2.5-Coder-32B-Instruct`: `Babel-translate` (Lean↔Rocq) and `Babel-ssreflect` (Rocq→SSReflect). We sample up to 128 candidates per goal at inference.
- **Direct translators.** Prompt GPT‑5 for script-to-script translation with interactive repair. Combining GPT‑5 and Babel gives the strongest results.

## Benchmarks & results
| Setup | Lean → Rocq | Rocq → Lean | Rocq → SSReflect |
| --- | --- | --- | --- |
| GPT‑5 (4 feedback rounds) | **82.9 %** | **67.5 %** | 16.6 % |
| Babel-translate (128 samples) | 68.3 % | 40.2 % | – |
| Babel-ssreflect (128 samples) | – | – | **33.2 %** |
| Babel w/o reasoning | – | – | 21.8 % |
| GPT‑5 + Babel (union) | 89.7 % | 83.7 % | 34 % |

## Repository map
- `src/proof_dumps/`: quick proof-term dumps for prefiltering (length, tokens) without running the full extraction pipeline.
- `src/lean_rocq_translation/step_*`: end-to-end pipeline for Lean↔Rocq extraction, filtering, prompting, and model inference.
- `src/evaluation/`: Lean and Rocq evaluators, configs, and prompt templates.
- `src/training_nemo/`: training jobs built on NVIDIA NeMo; tweak hyperparameters in `config/training/nemo.yaml` and launch with the SLURM scripts in `config/training/training_h100.slurm` (train) and `config/training/eval_h100.slurm` (eval).
- `dataset/`: aligned Lean/Rocq benchmark, SSReflect variants, and auxiliary corpora.
- `paper/`: full NeurIPS workshop submission.
- `doc/`: detailed pipeline notes (see `doc/details.md`).

## Getting started
```bash
conda create -n babel python==3.10
conda activate babel
pip install -r requirements.txt
pip install -e coqpyt
```
Install [Pytanque](https://github.com/LLM4Rocq/pytanque) for Rocq evaluation and repair loops.

## Running the pipelines
1. (Optional) Prefilter proofs by length with the scripts in `src/proof_dumps/` to shortlist the ones you want to process.
2. Extract Lean terms: `python src/lean_rocq_translation/step_1/exec.py --project /path/to/mathlib`.
3. Align terms with proofs and down-select: steps 2–4 in `src/lean_rocq_translation`.
4. Generate reasoning traces (step 5) and prompts for Babel fine-tuning (step 6).
5. Launch model sampling with `src/lean_rocq_translation/step_7/exec.py` or `src/lean_rocq_translation/step_7_rocq/exec.py`.
6. Evaluate candidates with `src/evaluation/lean_evaluate.py` and `src/evaluation/rocq_evaluate.py`.

Full step-by-step notes, command flags, and expected outputs are [here](doc/details.md).

## Data, models, and figures
Aligned Lean/Rocq pairs: `dataset/repo/rocq/*.v` and matching Lean files under `dataset/`.
