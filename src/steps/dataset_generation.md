# Dataset generation

A set of steps to generate the final datasets, see [Fine tuning of LLM section](/doc/details.md#fine-tuning-of-llm) for details, see [Fig.1.](#pipeline) for a summary.

* step_0: Extracts all terms, premises, notations, and proofs.
* step_0_bis: Checks that the previously parsed information is accurate by recompiling with Pytanque; extract goals and subgoals.
* step_1: Computes the lengths of tokenized lambda terms, and sequences of tactics.
* step_2: Filters datasets based on maximum/minimum number of tokens in terms.
* step_3: Extracts a subset of **diverse** terms by maximizing the BM25 distance of statements + proofs.
* step_4: For each term, generates multiple reasonings using **DeepSeek R1**.
* step_5: Computes the score for each reasoning based on [this](doc/details.md#sixth-step) method, leveraging a smaller model.
* step_6: Extracts a final reasoning for each term, starting from best ones (score-wise), and checks (using **O3-mini**) whether they do not rely on already knowing the sequence of tactics.

<center id="pipeline">

|![image](/doc/img/pipeline.png)|
|:--:|
|**Fig. 1.** Pipeline for data augmentation|
</center>

## Usage

To avoid issues with file collisions/corruption, aid debugging, and to have easily adaptable code, we implement a step-by-step pipeline.

### First step

Download [here](https://drive.proton.me/urls/2BHPS9QM0R#OEzClePu0yJ6) the result of this first step (and step_0_bis) if you don't want to compute it from scratch.

To generate terms, notations, constants etc. replace $MATHCOMP_PATH by a directory containing mathcomp (e.g., /home/username/.opam/default/.opam-switch/sources/ if you install it through Opam).

```console
python -m src.steps.step_0.exec --input $MATHCOMP_PATH --output export/step_0
```

You can start workers in parallel using idx-worker, and num-workers parameters: 
```console
python -m src.steps.step_0.exec --input $MATHCOMP_PATH --idx-worker 0 --num-workers 3 &\
python -m src.steps.step_0.exec --input $MATHCOMP_PATH --idx-worker 1 --num-workers 3 &\
python -m src.steps.step_0.exec --input $MATHCOMP_PATH --idx-worker 2 --num-workers 3
```

### First step bis
*Reminder: [here](https://drive.proton.me/urls/2BHPS9QM0R#OEzClePu0yJ6) is the already compute dataset*.

Checks that previously parsed information is accurate by recompiling using Pytanque, extract goals and subgoals.

```console
python -m src.steps.step_0_bis.exec --input export/step_0 --output export/step_0_bis
```

### Second step

Computes the length of tokenized lambda-terms, and sequences of tactics using the provided tokenizer.

```console
python -m src.steps.step_1.exec --input export/step_0_bis --output export/step_1 --tokenizer 'deepseek-ai/DeepSeek-Prover-V1.5-Base'
```

### Third step

Filters the previous data based on term length and number of steps in proofs, then selects a diverse subset using BM25.

```console
python -m src.steps.step_2.exec --input export/step_1 --output export/step_2 --num-documents 1000 --max-num-tokens 3750 --min-number-instructions 3 --max-number-instructions 7
```

### Fourth Step

Generate reasonings using DeepSeek R1.

You need to export a variable $OPENAI_API_KEY containing an API key. By default, OpenRouter is configured, change src/steps/step_3/config.yaml as needed.

Change max_workers to choose the number of parallel generations.
Each worker is assigned a prompt to generate a reasoning; worker waits between $0$, and $2\cdot mean\_delay$ seconds (uniform sampling) and then generates $num\_gen$ reasonings sequentially.

To avoid reaching API limits at the beginning of the script (all worker initialize at the same time), please adjust mean_delay (in seconds) parameter.
A rule of thumb is that the number of requests per second is approximately $\frac{max\_workers}{mean\_delay}$, please add a safety margin to avoid issues. 

Example for around 10 requests per second at the beginning (then less since generation is slower than mean_delay)
```console
python -m src.steps.step_3.exec --input export/step_2 --output export/step_3 --num_gen 20 --max_workers 100 --mean-delay 10
```

### Fifth Step

Compute the score for each reasoning based on prediction performance with DeepSeek-R1-Distill-Qwen-32B.
This step requires retrieving **logprobs results from the prompt**.
The base config is adapted to [VLLM](#https://docs.vllm.ai/en/latest/getting_started/quickstart.html).
Don't forget to change the base_url parameter in src/steps/step_4/config.yaml

```console
python -m src.steps.step_4.exec --input export/step_3 --output export/step_4
```

### Sixth Step

Filters the best reasonings by asking an LLM (O3-mini) to check additionnal constraints. Same recommendation as in the [fourth step](#fourth-step).

```console
python -m src.steps.step_3.exec --input export/step_4 --output export/step_5 --num_gen 20 --max_workers 100 --mean-delay 10
```

### Sixth Step

Export previous step into one json.

```console
python -m src.steps.step_3.exec --input export/step_4 --output export/step_5 --num_gen 20 --max_workers 100 --mean-delay 10
```
