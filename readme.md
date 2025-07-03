# Babel-Formal

The goal of this repository is to translate proofs between Rocq and Lean by leveraging proof terms.

More precisely, given two formal statements, one proved (source), one to be proved (target), and a list of matching premises, translate the proof from source to target.

Due to the difficulty of getting matching premises, a first task we consider is style transfer: converting Rocq proofs to **MathComp** style (i.e. leveraging the **SSReflect** set of tactics).

We experiment around the following questions:
* Train a **transformer model** to translate proof terms (from Lean or Rocq) into a sequence of tactics in the same language (*decompilation*).
* Train a **transformer model** to translate proofs from **vanilla Rocq** to **SSReflect**, based on proof terms (*style transfer*).
* Obtain a model capable of **translating tactics from one language to another** by leveraging proof terms from **one language** to generate a sequence of tactics **in another** (e.g., Lean to Rocq).

Our project leverages proof terms as an intermediate representation to translate proofs between Rocq and Lean.
To do so, we will fine-tune an LLM on this specific task.

**For more details, please read [this](doc/details.md).**


## Implementation

This repository consists of several components:

* An extension of CoqProof class from **CoqPyt**, which drops some features to make computation tractable on large files and also recovers proof term, constants and notations.
* A (small) extension to LeanDojo to recover proof terms and notations (**WIP**).
* A set of experimentation scripts, used to generate various figures, and to provide a glimpse of some features of the dataset.
* A set of steps to generate the final datasets, see [Fine tuning of LLM section](doc/details.md#fine-tuning-of-llm) for details.
* An evaluation script to compute performance (pass@k) of models.

## Dependencies

You should set up a virtual environment first, e.g., using miniconda and install the repository dependencies.

```console
conda create -n $ENV_NAME python==3.10
conda activate $ENV_NAME
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
