# Jean Zay Benchmarking

This document describes the new benchmark pipeline for Lean <-> Rocq translation.

## 1) Prepare benchmark files (SLURM + HF)

`src.benchmark.prepare` now integrates proof-term extraction automatically by default
(`--proof-term-extraction always`).
Extraction uses the same extraction family as training data:
- Lean: `src.lean_rocq_translation.step_4.exec::TheoremExtractor`
- Rocq: diagnostics-based `Print / Print All Dependencies / Check`

```bash
python -m src.benchmark.prepare \
  --benchmark-root new_benchmark \
  --output-dir benchmark/prepared \
  --num-shards 8
```

If you want extraction as a separate step:

```bash
python -m src.benchmark.extract_proof_terms \
  --benchmark-root new_benchmark
```

Quick debug on one file:

```bash
python -m src.benchmark.extract_proof_terms \
  --benchmark-root new_benchmark \
  --stems algebra_group_cancel
```

Optional cap (after `--stems` filtering):

```bash
python -m src.benchmark.extract_proof_terms \
  --benchmark-root new_benchmark \
  --max-stems 8
```

This writes:

- `new_benchmark/proof_terms/lean.json`
- `new_benchmark/proof_terms/rocq.json`
- `new_benchmark/proof_terms/manifest.json`

Accepted format for each file (dependencies are included when extracted):

```json
[
  {
    "source_stem": "algebra_group_cancel",
    "name": "thm_name",
    "term": "<canonical proof term>",
    "dependencies": [{"name": "dep", "fqn": "Lib.dep", "type": "<dep type>"}]
  }
]
```

or nested map:

```json
{
  "algebra_group_cancel": {
    "thm_name": "<canonical proof term>",
    "thm_name_2": "<canonical proof term>"
  }
}
```

The pipeline is strict:
- it raises an error if any proof term is missing
- it raises an error if Lean/Rocq extracted theorem sets do not match

Create flat JSONL files and optional array-job shards:

```bash
python -m src.benchmark.prepare \
  --benchmark-root new_benchmark \
  --output-dir benchmark/prepared \
  --num-shards 8
```

Optional extraction policy controls:
- `--proof-term-extraction always|auto|never`
- `--proof-term-rocq-timeout 180`
- `--proof-term-stems stem_a,stem_b`
- `--proof-term-max-stems 8`

Optional upload to Hugging Face dataset:

```bash
export HF_TOKEN=...
python -m src.benchmark.prepare \
  --benchmark-root new_benchmark \
  --output-dir benchmark/prepared \
  --num-shards 8 \
  --hf-repo-id your-org/babel-benchmark-prepared \
  --hf-private
```

Outputs:
- `benchmark/prepared/lean_to_rocq.jsonl`
- `benchmark/prepared/rocq_to_lean.jsonl`
- `benchmark/prepared/lean_to_rocq/shard_XXX.jsonl`
- `benchmark/prepared/rocq_to_lean/shard_XXX.jsonl`

## 2) Run benchmark directly

```bash
python -m src.benchmark.run \
  --direction both \
  --input benchmark/prepared/benchmark_pairs.json \
  --model-path /path/to/model_or_hf_id \
  --pass-k 32 \
  --max-rounds 2 \
  --feedback-level goals \
  --temperature 0.7 \
  --top-p 0.95 \
  --output-dir export/benchmark_runs
```

### Important parameters
- `--direction`: `lean_to_rocq`, `rocq_to_lean`, `both`
- `--pass-k`: number of sampled candidates
- `--max-rounds`: retry rounds with feedback per candidate
- `--feedback-level`: `whole_proof`, `goals`, `errors`, `goals_errors`, `no_feedback`
- `--disable-reasoning`: no-reasoning prompt mode
- `--model-path`: accepts HF id or local dir; NeMo-style dirs are auto-resolved to a HF checkpoint if possible

## 3) Lean backend (Kimina)

Rocq -> Lean checking uses Kimina (no local Lean backend in `src.benchmark.run`):

```bash
export LEAN_SERVER_API_KEY=...   # optional if your server is protected
python -m src.benchmark.run \
  --direction rocq_to_lean \
  --model-path /path/to/model \
  --lean-kimina-url http://127.0.0.1:8000 \
  --lean-kimina-timeout 300
```

## 4) Rocq backend (`rocq-ml-server`)

```bash
python -m src.benchmark.run \
  --direction lean_to_rocq \
  --model-path /path/to/model \
  --rocq-backend ml_server \
  --rocq-host 127.0.0.1 \
  --rocq-port 5000
```

## 5) SLURM jobs

### Standard benchmark job

`config/benchmark/benchmark_jz.slurm`

Example:

```bash
sbatch --export=ALL,\
MODEL_PATH=/path/to/model,\
DIRECTION=both,\
PASS_K=64,\
MAX_ROUNDS=2,\
FEEDBACK_LEVEL=goals,\
LEANKIMINA_URL=http://127.0.0.1:8000,\
ROCQ_BACKEND=ml_server,\
ROCQ_PORT=5000 \
config/benchmark/benchmark_jz.slurm
```

If using a singularity/apptainer image, also export `SIF_IMAGE=/path/to/image.sif`.

### Ablation array job

`config/benchmark/benchmark_ablation_jz.slurm`

1) copy the example matrix:

```bash
cp config/benchmark/model_matrix.example.txt config/benchmark/model_matrix.txt
```

2) set rows to your models (`label|model_path|disable_reasoning`)

3) run array:

```bash
sbatch --array=0-3 \
  --export=ALL,MODEL_MATRIX_FILE=config/benchmark/model_matrix.txt,DIRECTION=both,PASS_K=32 \
  config/benchmark/benchmark_ablation_jz.slurm
```

This is intended for:
- base model vs fine-tuned model
- reasoning prompt on/off
- no-reasoning fine-tuned checkpoints

## 6) Output artifacts

Each run writes:
- `run_config.json`: resolved config
- `attempts_<direction>.jsonl`: all attempts (raw model output, extracted proof, verifier feedback)
- `results_<direction>.jsonl`: per-theorem success/failure
- `summary.json`: aggregated pass@k
