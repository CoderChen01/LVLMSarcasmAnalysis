import base64
import importlib
import importlib.resources
import json
import os
import time
from io import BytesIO
from typing import Any, Literal, Optional, Union

import httpx
import tomli
from loguru import logger
from openai import (
    NOT_GIVEN,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    NotGiven,
    OpenAI,
)
from PIL.ImageFile import ImageFile
from pydantic import BaseModel, Field


def image_to_base64(image: ImageFile) -> str:
    with BytesIO() as buffered:
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class BSCTaskOutput(BaseModel):
    classification: Literal["sarcastic", "non-sarcastic"]
    rationale: str
    score: float


class TSCTaskOutput(BaseModel):
    classification: Literal["sarcastic", "non-sarcastic", "neutral"]
    rationale: str
    score: float


class xCSTaskOutput(BaseModel):
    # SCS Task or LCS Task
    rationale: str
    score: float


class CompletionParamBuilder:

    def __init__(self, task_name: str, seed: int) -> None:
        self._task_name = task_name
        self._seed = seed

        self._num_words = 150
        self._temperature = 0.0

        self.all_prompts: dict[str, str] = tomli.loads(
            importlib.resources.read_text(
                "lvlm_sarcasm_analysis.prompts", f"{self._task_name}.toml"
            )
        )["prompts"]

        if self._task_name == "bsc-task":
            self._data_model = BSCTaskOutput
        elif self._task_name in [
            "scs-task",
            "lcs-task",
        ]:
            self._data_model = xCSTaskOutput
        elif self._task_name == "tsc-task":
            self._data_model = TSCTaskOutput
        else:
            raise ValueError("Invalid prompt mode")

    @property
    def prompt_variants(self) -> list[str]:
        return list(self.all_prompts.keys())

    def _get_system_prompt(self, prompt_variant: str) -> str:
        return self.all_prompts[prompt_variant].format(x=self._num_words)

    def __call__(
        self, text: str, image: ImageFile, model: str, prompt_name: str
    ) -> dict:
        image_base64 = image_to_base64(image)
        image_data = f"data:image/jpeg;base64,{image_base64}"

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self._get_system_prompt(prompt_name),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": text},
                ],
            },
        ]

        self._num_words -= 15
        if self._num_words <= 0:
            self._temperature += 0.1
            if self._temperature >= 1.0:
                self._temperature = 1.0
            if self._temperature == 1.0:
                self._seed += 10
            self._num_words = 150

        return {
            "model": model,
            "messages": messages,
            "temperature": self._temperature,
            "seed": self._seed,
            "logprobs": True,
            "response_format": self._data_model,
        }


def get_api_key(keys: list[str], exclude_keys: Optional[list[str]] = None) -> str:

    if exclude_keys is not None:
        new_keys = [key for key in keys if key not in exclude_keys]

        if len(new_keys) == 0:
            new_keys = keys

        return new_keys[0]

    return keys[0]


def get_base_url(
    base_urls: dict[str, list[str]],
    model: str,
    excluded_base_urls: Optional[list[str]] = None,
) -> str:
    urls = base_urls[model]

    if excluded_base_urls is not None:
        new_urls = [url for url in urls if url not in excluded_base_urls]

        if len(new_urls) == 0:
            new_urls = urls

        return new_urls[0]

    return urls[0]


def get_config(path: str, key: str, **kwargs) -> Any:

    success_load = False
    while not success_load:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            success_load = True
        except Exception:
            logger.exception("Error in reading config file [{}]", path)
            time.sleep(1 / 10)

    if key == "api_keys":
        keys: list[str] = data[key]
        return get_api_key(keys, **kwargs)
    elif key == "base_urls":
        base_urls: dict[str, list[str]] = data[key]
        return get_base_url(base_urls, **kwargs)

    return data[key]


def requests_map_func(
    examples,
    mode: str,
    config_file_path: str,
    model: str = "Qwen/Qwen2-VL-7B-Instruct",
    seed: int = 42,
) -> dict:

    if mode == "vllm":
        base_url = get_config(config_file_path, "base_urls", model=model)
        client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=0)
        excluded_base_urls = set()
    elif model == "openai":
        api_key = get_config(config_file_path, "api_keys")
        client = OpenAI(api_key=api_key)
        exclude_keys = set()
    else:
        raise ValueError("Invalid mode")

    task_names: list[str] = get_config(config_file_path, "metadata")["task_names"]

    for task_name in task_names:
        for i in range(len(examples["text"])):
            image = examples["image"][i]
            text = examples["text"][i]
            req_builder = CompletionParamBuilder(task_name, seed)

            for prompt_variant in req_builder.prompt_variants:
                req_builder = CompletionParamBuilder(task_name, seed)

                is_reset = False
                if prompt_variant.endswith("_reset"):
                    is_reset = True

                have_response = False
                reponse_key_name = (
                    f"{model}__{task_name}__{prompt_variant.replace('_reset', '')}"
                )
                if reponse_key_name not in examples:
                    examples[reponse_key_name] = []
                else:
                    have_response = True

                req = req_builder(text, image, model, prompt_variant)

                if (
                    have_response
                    and i < len(examples[reponse_key_name])
                    and examples[reponse_key_name][i] != ""
                    and not is_reset
                ):
                    logger.info("Already have response for [{}]", examples["id"][i])
                    continue

                is_request = True
                success = False
                error_count = 0
                while not success:
                    try:
                        if is_request:
                            with client.beta.chat.completions.stream(
                                **req,
                                timeout=60,
                                max_tokens=512,
                                extra_body={"repetition_penalty": 1.2},
                            ) as st:
                                res = st.get_final_completion()
                            res = res.to_json()
                            if have_response and (
                                i < len(examples[reponse_key_name]) or is_reset
                            ):
                                examples[reponse_key_name][i] = res
                            else:
                                examples[reponse_key_name].append(res)
                            logger.success(
                                "{}__{}__{}__{}__{}",
                                task_name,
                                prompt_variant,
                                examples["id"][i],
                                examples["label"][i],
                                res,
                            )
                            success = True
                        else:
                            is_request = get_config(config_file_path, "metadata")[
                                "is_request"
                            ]
                            logger.info("Request paused for [{}]", examples["id"][i])
                            time.sleep(10)
                    except (
                        LengthFinishReasonError,
                        httpx.ReadTimeout,
                        ValueError,
                        ContentFilterFinishReasonError,
                    ):
                        logger.warning(
                            "Length finish reason error for [{}]", examples["id"][i]
                        )
                        req = req_builder(text, image, model, prompt_variant)
                        logger.info("Retry with [{}]", req)
                    except Exception:
                        is_request = get_config(config_file_path, "metadata")[
                            "is_request"
                        ]

                        error_count += 1
                        logger.exception("Error in request [{}]", examples["id"][i])
                        time.sleep(5)

                        if error_count <= 3:
                            error_count += 1
                            continue

                        if mode == "vllm":
                            logger.warning("Too many errors, change base url")
                            old_exclude_base_urls_len = len(excluded_base_urls)
                            excluded_base_urls.add(base_url)
                            if old_exclude_base_urls_len == len(excluded_base_urls):
                                excluded_base_urls.clear()

                            # if error_count > 10:
                            base_url = get_config(
                                config_file_path,
                                "base_urls",
                                model=model,
                                excluded_base_urls=list(excluded_base_urls),
                            )
                            client = OpenAI(
                                api_key="EMPTY", base_url=base_url, max_retries=0
                            )
                        elif mode == "openai":
                            logger.warning("Too many errors, changing API key")
                            old_exclude_key_len = len(exclude_keys)
                            exclude_keys.add(api_key)
                            if old_exclude_key_len == len(exclude_keys):
                                # error_count += 1
                                exclude_keys.clear()

                            api_key = get_config(
                                config_file_path,
                                "api_keys",
                                exclude_keys=list(exclude_keys),
                            )
                            client = OpenAI(api_key=api_key)
                        continue
    return examples


def local_data_map_func(
    examples,
    local_data: Optional[dict[str, dict]] = None,
) -> dict:
    if local_data is None:
        return examples

    for response_name in local_data:
        have_response = False

        if response_name not in examples:
            examples[response_name] = []
        else:
            have_response = True

        for i in range(len(examples["text"])):
            id_ = examples["id"][i]
            if have_response and examples[response_name][i] != "":
                logger.info("Already have response for [{}]", id_)
                continue
            if id_ in local_data[response_name]:
                if have_response:
                    examples[response_name][i] = json.dumps(
                        local_data[response_name][id_]
                    )
                else:
                    examples[response_name].append(
                        json.dumps(local_data[response_name][id_])
                    )
                logger.info(f"Using local data for {response_name}[{id_}]")
                continue

            logger.info(f"No local data for {response_name}[{id_}]")

            if not have_response:
                examples[response_name].append("")

    return examples
