# Training and Evaluation

**WIP**

## Usage

### Generation

See [Fourth step](/src/steps/dataset_generation.md#fourth-step) for details about max-workers and mean-delay parameter

```
python -m src.experiment.exp_0.exec --input-dataset export/steps/final.json \
--input-sources export/steps/sources \
--output export/eval \
--dataset-entry test \
--k 4 \
--config src/training/config/o3minihigh.yaml \
--max-workers 100 \
--mean-delay 10
```

### Evaluation

```
python -m src.experiment.exp_0.exec --input export/eval_r1 --input-sources export/steps/sources --output export/eval_r1.json
```

