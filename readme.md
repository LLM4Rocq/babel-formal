# Babel-Formal

This repository has two goals
* Train a **transformer model** to translate lambda-terms (from Lean or Rocq) into a sequence of tactics in the same language.
* Obtain a model capable of translating tactics from one language to another by leveraging lambda-terms from one language to generate a sequence of tactics in a different language.

Our project leverages $\lambda$-calculus as an intermediate representation to translate proofs between Rocq and Lean.
To do so, we will fine-tune an LLM on this specific task.

**For more details, please read [this](doc/details.md).**


## Implementation

This repository consists of several components:

* An extension of CoqProof class from **CoqPyt**, which drops some features to make computation tractable on large files and also recovers $\lambda$-terms, constants and notations.
* A set of experimentation scripts, used to generate various figures, and to provide a glimpse of some features of the dataset.
* A set of steps to generate the final datasets, see [Fine tuning of LLM section](doc/details.md#fine-tuning-of-llm) for details.
* An evaluation script to compute performance of models.

## Dependencies

You should set up a virtual environment first, e.g., using miniconda and install the repository dependencies.

```console
conda create -n env_name python==3.10
pip install -r requirements.txt
pip install -e coqpyt
```

Additionally, this repository requires the installation of [Pytanque](https://github.com/LLM4Rocq/pytanque) and Petanque (see [Pytanque repo](https://github.com/LLM4Rocq/pytanque)).

## Usage

### Experimentation

See [here](/src/experiments/experiments.md).

### Dataset generation

See [here](/src/steps/dataset_generation.md) to use the code, or download directly the dataset [here](https://drive.proton.me/urls/MDAERQJD0C#D3DFuDCDXmNU).

### Training & Evaluation

See [here](/src/training/training_eval.md).
