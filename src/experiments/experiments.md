# Experimentations

A set of experimentation scripts, used to generate various figures, and to provide a glimpse of some features of the dataset.

* exp_0: Histogram of the lengths of tokenized terms, and sequences of tactics (requires completion of step_1).
* exp_1: Scores of different reasonings for a given term (requires completion of step_4).
* exp_2: Histogram of different reasoning scores; to be used when there is a more diverse set of reasonings (requires completion of step_4).
* exp_3: Comparison between rankings obtained from LLM calls and reasoning scores.
* exp_4: Result from evaluation of LLMs on the task of translation (requires evaluation).
* exp_5: Scaling law associated evaluation of LLMs on the task of translation (requires evaluation).
* exp_6: Accuracy vs Original Proof Length (requires evaluation).
* exp_7: Accuracy vs Generated Proof Length (requires evaluation).

## Usage

### Experiment 0: Histogram of tokenized input
Required step_1 to be completed (see [here](/src/steps/dataset_generation.md#second-step)).

```
python -m src.experiment.exp_0.exec --input export/steps/step_1 --output export/experiment/exp_0
```

### Experiments 1-3:

Need clean up.

### Experiment 4: Evaluation
Required evaluation to be done (see [WIP](/src/training/training_eval.md)).

```
python -m src.experiment.exp_4.exec --input export/evaluation/result --output export/experiment/exp_4
```

### Experiment 5: Scaling law
Required evaluation to be done (see [WIP](/src/training/training_eval.md)).

```
python -m src.experiment.exp_5.exec --input export/evaluation/result --output export/experiment/exp_5
```

### Experiment 6: Accuracy vs Original Proof Length
Required evaluation to be done (see [WIP](/src/training/training_eval.md)).

```
python -m src.experiment.exp_6.exec --input export/evaluation/result --output export/experiment/exp_6 --source export/benchmark.json
```

### Experiment 7: Accuracy vs Generated Proof Length
Required evaluation to be done (see [WIP](/src/training/training_eval.md)).

```
python -m src.experiment.exp_7.exec --input export/evaluation/result --output export/experiment/exp_7
```
