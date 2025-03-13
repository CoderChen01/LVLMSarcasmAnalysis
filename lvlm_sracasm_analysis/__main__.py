import datetime
import json
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, Type, cast

import jsonlines
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(
    lambda msg: tqdm.write(msg, end=""),
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    enqueue=True,
    colorize=True,
)
curent_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logger.add(
    f"run_sarcasm_bench_{curent_datetime}.log",
    serialize=True,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    enqueue=True,
)

import click

from sarcbench.generator_map_functions import local_data_map_func, requests_map_func


@click.group(chain=True, invoke_without_command=False)
@click.option("--dataset-path", type=str, help="Dataset to use")
@click.option("--dataset-name", type=str, required=False, help="Dataset name")
@click.option("--dataset-split", type=str, required=False, help="Dataset split")
@click.option("--config-file-path", type=str, help="Config file path")
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output path",
)
@click.option(
    "--num-debug-samples",
    type=int,
    default=-1,
    help="Number of debug samples (for testing)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for reproducibility",
)
def run_sarcasm_bench(
    dataset_path: str,
    dataset_name: Optional[str],
    dataset_split: Optional[str],
    config_file_path: str,
    output_path: Path,
    num_debug_samples: int,
    seed: int,
) -> None:
    pass


@run_sarcasm_bench.result_callback()
def process_pipeline(
    processors,
    dataset_path: str,
    dataset_name: str,
    dataset_split: str,
    config_file_path: str,
    output_path: Path,
    num_debug_samples: int,
    seed: int,
):
    try:
        datasets = load_dataset(dataset_path, dataset_name, split=dataset_split)
    except ValueError:
        datasets = load_from_disk(dataset_path)

    if isinstance(datasets, DatasetDict):
        datasets = datasets
    else:
        datasets = {"default": datasets}

    finished_datasets = {}
    for key, dataset in datasets.items():
        dataset = cast(Dataset, dataset)
        logger.info(f"evaluating dataset ({dataset_name}, {key})")

        if num_debug_samples > 0:
            n = dataset.filter(lambda x: x["label"] == 0)
            p = dataset.filter(lambda x: x["label"] == 1)
            dataset = concatenate_datasets(
                [
                    n.select(range(num_debug_samples // 2)),
                    p.select(range(num_debug_samples // 2)),
                ]
            )

        for processor in processors:
            dataset = processor(dataset, config_file_path, seed)

        finished_datasets[key] = dataset

    output_path.mkdir(parents=True, exist_ok=True)
    for key, dataset in finished_datasets.items():
        save_path = (
            output_path
            / f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{key}"
        )
        dataset.save_to_disk(save_path)
        logger.info(f"saved to {save_path}")


@run_sarcasm_bench.command("openai", help="Run OpenAI processor")
@click.option("--model", type=str, default="gpt-4o", help="Model to use")
@click.option("--num-proc", type=int, default=-1, help="Number of processes to use")
def run_openai(
    model: str,
    num_proc: int,
):

    if num_proc <= 0:
        cpucount = os.cpu_count()
        if cpucount is None:
            cpucount = 16
        num_proc = int(cpucount * 0.8)
    logger.info(f"num_proc: {num_proc}")

    def processor(dataset, config_path, seed):
        dataset = dataset.map(
            partial(
                requests_map_func,
                mode="openai",
                config_file_path=config_path,
                model=model,
                seed=seed,
            ),
            batched=True,
            num_proc=num_proc,
        )
        dataset.info.description = f"{dataset.info.description}+{model}"
        return dataset

    return processor


@run_sarcasm_bench.command("vllm", help="Run VLLM processor")
@click.option(
    "--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model to use"
)
@click.option("--num-proc", type=int, default=-1, help="Number of processes to use")
def run_vllm(
    model: str,
    num_proc: int,
):

    if num_proc <= 0:
        cpucount = os.cpu_count()
        if cpucount is None:
            cpucount = 16
        num_proc = int(cpucount * 0.9)
    logger.info(f"num_proc: {num_proc}")

    def processor(dataset, config_path, seed):
        dataset = dataset.map(
            partial(
                requests_map_func,
                mode="vllm",
                config_file_path=config_path,
                model=model,
                seed=seed,
            ),
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=False,
        )
        dataset.info.description = f"{dataset.info.description}+{model}"
        return dataset

    return processor


def extract_log(log_path: str) -> dict:
    all_data = defaultdict(dict)

    reader = jsonlines.open(log_path)
    for obj in reader:
        level = obj["record"]["level"]["name"]
        if level != "SUCCESS":
            continue

        message: str = obj["record"]["message"]
        prompt_mode, prompt_name, id_, _, data = message.split("__", maxsplit=4)
        response: dict[str, Any] = json.loads(data)

        all_data[f"{response['model']}__{prompt_mode}__{prompt_name}"][id_] = response

    return all_data


@run_sarcasm_bench.command("write-log-to-dataset", help="Write log to dataset")
@click.option("--log-path", "-l", type=str, help="Log file path")
def write_log_to_dataset(
    log_path: str,
):

    all_response_data = extract_log(log_path)

    def processor(dataset, config_path, seed):
        dataset = dataset.map(
            partial(local_data_map_func, local_data=all_response_data),
            batched=True,
            num_proc=1,  # No blocking requests to the server, so no need for multiple processes such that we can save memory
        )
        return dataset

    return processor


@click.command()
@click.option("--config-path", "-c", type=str, help="Config path")
@click.option("--data-path", "-d", type=str, help="Data path")
@click.option("--output-path", "-o", type=str, help="Output path")
@click.option("--recompute", "-r", is_flag=True, help="Recompute")
@click.option(
    "--analysis",
    "-A",
    type=click.Choice(["agreement_gt", "model_nll", "neutral_label", "inter_prompt"]),
    multiple=True,
)
def analysizer(
    config_path: str,
    data_path: str,
    output_path: str,
    recompute: bool,
    analysis: list[
        Literal["agreement_gt", "model_nll", "neutral_label", "inter_prompt"]
    ],
) -> None:
    with open(config_path, "r") as f:
        META_DATA = json.load(f)

    from sarcbench.analysizer import (
        BaseReporter,
        GroundTruthAgreementReporter,
        ModelNllReporter,
        NeutralLabelOverlapReporter,
        PromptVariantConsistencyReporter,
    )

    CLS_MAP: dict[str, Type[BaseReporter]] = {
        "agreement_gt": GroundTruthAgreementReporter,
        "model_nll": ModelNllReporter,
        "neutral_label": NeutralLabelOverlapReporter,
        "inter_prompt": PromptVariantConsistencyReporter,
    }

    for reporter_cls_key in analysis:
        logger.info(f"Running {reporter_cls_key}")
        reporter_cls = CLS_MAP[reporter_cls_key]
        reporter = reporter_cls(
            data_path=data_path,
            output_path=output_path,
            model_group_info=META_DATA["model_groups"],
            model_name_alias=META_DATA["model_name_alias"],
            sorted_model_names=META_DATA["sorted_model_names"],
            is_recompute=recompute,
        )
        reporter.draw()
        logger.info(f"Finished {reporter_cls_key}")


if __name__ == "__main__":
    run_sarcasm_bench()
