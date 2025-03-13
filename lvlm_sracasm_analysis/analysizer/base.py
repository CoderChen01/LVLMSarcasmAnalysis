import json
import textwrap
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, cast

import pandas as pd
from datasets import Dataset, load_from_disk
from PIL.ImageFile import ImageFile

BERT_SCORE_MODEL = "microsoft/deberta-xlarge-mnli"
BERT_SCORE_BATCH_SIZE = 128
BERT_SCORE_PARTITION_SIZE = 1000


class BaseModelOutput(ABC):

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict, logprobs: list[dict]) -> "BaseModelOutput":
        pass

    @classmethod
    def from_response(cls, json_str: str) -> "BaseModelOutput":
        resp = json.loads(json_str)
        data: dict = resp["choices"][0]["message"]["parsed"]
        logprobs: list[dict] = resp["choices"][0]["logprobs"]["content"]

        return cls.from_json(data, logprobs)

    @staticmethod
    def calc_nll(logprobs: list[dict]) -> float:
        return sum([-logprob["logprob"] for logprob in logprobs]) / len(logprobs)


@dataclass
class xCSTaskModelOutput(BaseModelOutput):
    score: float
    rationale: str
    nll: float
    is_successful: bool

    @classmethod
    def from_json(cls, data: dict, logprobs: list[dict]) -> "xCSTaskModelOutput":
        score, rationale = data["score"], data["rationale"]
        nll = cls.calc_nll(logprobs)

        is_successful = True
        if score is None or score < 0 or score > 1:
            is_successful = False

        return cls(
            score=score,
            rationale=rationale,
            nll=nll,
            is_successful=is_successful,
        )

    def __str__(self) -> str:
        return f"Score={self.score}\nRationale={textwrap.fill(self.rationale, width=100)}\nNLL={self.nll}\nIsSuccessful={self.is_successful}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class xSCTaskModelOutput(BaseModelOutput):
    score: float
    rationale: str
    nll: float
    classification: str
    is_successful: bool

    prediction_label: int = field(init=False)

    @classmethod
    def from_json(cls, data: dict, logprobs: list[dict]) -> "xSCTaskModelOutput":
        score, rationale, classification = (
            data["score"],
            data["rationale"],
            data["classification"],
        )
        nll = cls.calc_nll(logprobs)

        is_successful = True
        if score is None or score < 0 or score > 1:
            is_successful = False

        return cls(
            score=score,
            rationale=rationale,
            nll=nll,
            is_successful=is_successful,
            classification=classification,
        )

    def __post_init__(self) -> None:
        self.prediction_label = {"sarcastic": 1, "non-sarcastic": 0, "neutral": 2}[
            self.classification
        ]

    def __str__(self) -> str:
        return f"Classification={self.classification}\nScore={self.score}\nRationale={textwrap.fill(self.rationale, width=50)}\nNLL={self.nll}\nIsSuccessful={self.is_successful}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class SampleTaskOutputInfo:
    id: str
    human_label: int
    text: str
    image: ImageFile
    bsc_task_model_outputs: dict[str, xSCTaskModelOutput]
    tsc_task_model_outputs: dict[str, xSCTaskModelOutput]
    scs_task_model_outputs: dict[str, xCSTaskModelOutput]
    lcs_task_model_outputs: dict[str, xCSTaskModelOutput]


def get_all_sample_task_output_info(
    dataset: Dataset,
) -> dict[str, list[SampleTaskOutputInfo]]:
    all_exp_names = [c for c in dataset.column_names if "__" in c]

    human_label_map = OrderedDict()
    text_map = OrderedDict()
    image_map = OrderedDict()
    all_bsc_task_model_outputs = defaultdict(lambda: defaultdict(OrderedDict))
    all_tsc_task_model_outputs = defaultdict(lambda: defaultdict(OrderedDict))
    all_scs_task_model_outputs = defaultdict(lambda: defaultdict(OrderedDict))
    all_lcs_task_model_outputs = defaultdict(lambda: defaultdict(OrderedDict))

    model_names = set()
    for d in dataset:
        d = cast(dict, d)
        id = d["id"]
        human_label_map[id] = d["label"]
        text_map[id] = d["text"]
        image_map[id] = d["image"]

        for exp_name in all_exp_names:
            model_name, task_name, prompt_variant = exp_name.split("__")
            model_names.add(model_name)
            if task_name == "bsc-task":
                prediction = xSCTaskModelOutput.from_response(d[exp_name])
                all_bsc_task_model_outputs[model_name][id][prompt_variant] = prediction

            elif task_name == "tsc-task":
                prediction = xSCTaskModelOutput.from_response(d[exp_name])
                all_tsc_task_model_outputs[model_name][id][prompt_variant] = prediction

            elif task_name == "scs-task":
                prediction = xCSTaskModelOutput.from_response(d[exp_name])
                all_scs_task_model_outputs[model_name][id][prompt_variant] = prediction

            elif task_name == "lcs-task":
                prediction = xCSTaskModelOutput.from_response(d[exp_name])
                all_lcs_task_model_outputs[model_name][id][prompt_variant] = prediction
            else:
                raise ValueError(f"Unknown prompt mode: {task_name}")

    all_sample_task_output_info = defaultdict(list)
    for model_name in model_names:
        for id in human_label_map:
            all_sample_task_output_info[model_name].append(
                SampleTaskOutputInfo(
                    id=id,
                    human_label=human_label_map[id],
                    text=text_map[id],
                    image=image_map[id],
                    bsc_task_model_outputs=all_bsc_task_model_outputs[model_name][id],
                    tsc_task_model_outputs=all_tsc_task_model_outputs[model_name][id],
                    scs_task_model_outputs=all_scs_task_model_outputs[model_name][id],
                    lcs_task_model_outputs=all_lcs_task_model_outputs[model_name][id],
                )
            )

    return all_sample_task_output_info


class BaseReporter(ABC):
    name: ClassVar[str] = ""

    def __init__(
        self,
        data_path: str,
        model_name_alias: dict[str, str],
        model_group_info: dict[str, list[str]],
        sorted_model_names: list[str],
        output_path: str,
        is_recompute: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.output_path = Path(output_path) / self.name
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cached_results_path = self.output_path / "cached_results"
        self.cached_results_path.mkdir(parents=True, exist_ok=True)

        self.is_recompute = is_recompute

        self.model_name_alias = model_name_alias
        self.model_group_info = model_group_info
        self.sorted_model_names = sorted_model_names

        self.results = None

        self.load_cached_results()

        if self.results is None or self.is_recompute:
            dataset = load_from_disk(data_path)
            dataset = cast(Dataset, dataset)
            self.all_sample_task_output_info = get_all_sample_task_output_info(dataset)
            self.results = self.get_results()
            self.cache_results()

    @abstractmethod
    def get_results(self) -> dict[str, pd.DataFrame]:
        pass

    @abstractmethod
    def draw(self) -> None:
        pass

    def cache_results(self) -> None:
        if self.results is None:
            return

        for key, value in self.results.items():
            value.to_csv(str(self.cached_results_path / f"{key}.csv"))

    def load_cached_results(self) -> None:
        results = {}
        for file in self.cached_results_path.glob("*.csv"):
            key = file.stem
            results[key] = pd.read_csv(file, index_col=0)
        if len(results) > 0:
            self.results = results

    def get_model_name_alias(self, model_name: str) -> str:
        return self.model_name_alias.get(model_name, model_name)

    def _sort_model_names(self, df: pd.DataFrame) -> pd.DataFrame:
        sorted_model_names = []
        for n in self.sorted_model_names:
            if n in df.index:
                sorted_model_names.append(n)
        for n in df.index:
            if n not in sorted_model_names:
                sorted_model_names.append(n)
        df = df.loc[sorted_model_names]
        df.index = df.index.map(self.get_model_name_alias)
        return df
