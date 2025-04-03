# 🎉 Seeing Sarcasm Through Different Eyes: Analyzing Multimodal Sarcasm Perception in Large Vision-Language Models

---

[![arXiv](https://img.shields.io/badge/arXiv-2503.12149-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2503.12149)

With the advent of large vision-language models (LVLMs) demonstrating increasingly human-like abilities, a pivotal question emerges: do different LVLMs interpret multimodal sarcasm differently, and can a single model grasp sarcasm from multiple perspectives like humans? To explore this, we introduce an analytical framework using systematically designed prompts on existing multimodal sarcasm datasets. Evaluating 12 state-of-the-art LVLMs over 2,409 samples, we examine interpretive variations within and across models, focusing on confidence levels, alignment with dataset labels, and recognition of ambiguous ``neutral'' cases. Our findings reveal notable discrepancies---across LVLMs and within the same model under varied prompts. While classification-oriented prompts yield higher internal consistency, models diverge markedly when tasked with interpretive reasoning. These results challenge binary labeling paradigms by highlighting sarcasm’s subjectivity. We advocate moving beyond rigid annotation schemes toward multi-perspective, uncertainty-aware modeling, offering deeper insights into multimodal sarcasm comprehension.

<center>
<img src="assets/overview.svg" alt="framework-overview"/>
</center>

---

## ℹ️ Installation

```bash
poetry install
```

> If you don't install `pipx` and `poetry` yet, recommend to install them first.
>
> ```bash
> python3 -m pip install --user pipx
> python3 -m pipx ensurepath
> ```
>
> Then, install `poetry` via `pipx`.
>
> ```bash
> pipx install poetry
> ```
>
> You also can follow the official installation guide: [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation).

## 🕹 Evaluation

For each model evaluation, perform the following operations.

### 🤖 Start the OpenAI Compatable Server for the specific model

```bash
vllm serve <hf-model-id>  --task generate --trust-remote-code  --limit-mm-per-prompt image=1
```

### 🪄 Run the evaluation script

```bash
lvlm-sarc-evaluator --dataset-path  <path> --dataset-name <name [optional]> --dataset-split <split-name [optional]>  --output-path <output-path>  --config-file-path <config-path>  vllm --model  <hf-model-id>   --num-proc <num-proc>
```

> **Config File**
> 
> We introduced configuration files to control the behavior during the operation of `lvlm-sarc-evaluator` to improve fault tolerance, such as dynamically configuring the api_url and api_key of each model, and dynamically starting and pausing evaluation requests.
> [examples/evaluator_config.json](examples/evaluator_config.json) is an example of this configuration file, you can use it directly by specifying it directly through `--config-path`.

## 📉 Analysis

After performing evaluation for each model, we can get the final dataset and execute the following instructions to reproduce the results in our paper.

### Inter-Prompt Consistency Analysis

```bash
lvlm-sarc-analyzer --data-path <final-dataset-path> --output-path <result-output-path> --config-path <config-path> -A inter_prompt
```

### Agreement with Ground Truth Analysis

```bash
lvlm-sarc-analyzer --data-path <final-dataset-path> --output-path <result-output-path> --config-path <config-path> -A agreement_gt
```

### Model Confidence Analysis

```bash
lvlm-sarc-analyzer --data-path <final-dataset-path> --output-path <result-output-path> --config-path <config-path> -A model_nll
```

### Neutral Label Aalysis

```bash
lvlm-sarc-analyzer --data-path <final-dataset-path> --output-path <result-output-path> --config-path <config-path> -A neutral_label
```

> **Config File**
> 
> `lvlm-sarc-analyzer` will automatically draw the data graph. However, due to the long model name, the chart layout is not good, so we introduced a configuration file to configure short name for the model name to facilitate better layout of the data graph.
> [examples/analyzer_config.json](examples/analyzer_config.json) is an example of this configuration file, you can use it directly by specifying it directly through `--config-path`.
