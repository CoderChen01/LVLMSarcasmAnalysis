import gc
from collections import defaultdict
from typing import cast

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from torchmetrics.functional.text import bert_score

from .base import BaseReporter, SampleTaskOutputInfo

BERT_SCORE_MODEL = "microsoft/deberta-xlarge-mnli"
BERT_SCORE_BATCH_SIZE = 128
BERT_SCORE_PARTITION_SIZE = 1000


def prompt_variant_cls_consistency_analysis(
    all_sample_prediction_info: dict[str, list[SampleTaskOutputInfo]],
) -> pd.DataFrame:
    model_names = []
    results = []
    for model_name, sample_prediction_infos in all_sample_prediction_info.items():
        logger.debug("[CLS] Processing model: {}", model_name)

        direct_cls_preds = defaultdict(list)
        sarc_cls_preds = defaultdict(list)
        non_sarc_cls_preds = defaultdict(list)
        neutral_cls_preds = defaultdict(list)

        for sample in sample_prediction_infos:
            # BSC
            for (
                prompt_name,
                prediction,
            ) in sample.bsc_task_model_outputs.items():
                direct_cls_preds[prompt_name].append(prediction.prediction_label)

            # TSC
            for (
                prompt_name,
                prediction,
            ) in sample.tsc_task_model_outputs.items():
                neutral_cls_preds[prompt_name].append(prediction.prediction_label)

            # SCI
            for (
                prompt_name,
                prediction,
            ) in sample.scs_task_model_outputs.items():
                if prediction.is_successful:
                    if prediction.score > 0.5:
                        sarc_cls_preds[prompt_name].append(1)
                    else:
                        sarc_cls_preds[prompt_name].append(0)
                else:
                    sarc_cls_preds[prompt_name].append(np.nan)

            # LCI
            for (
                prompt_name,
                prediction,
            ) in sample.lcs_task_model_outputs.items():
                if prediction.is_successful:
                    if prediction.score < 0.5:
                        non_sarc_cls_preds[prompt_name].append(1)
                    else:
                        non_sarc_cls_preds[prompt_name].append(0)
                else:
                    non_sarc_cls_preds[prompt_name].append(np.nan)

        bsc_alpha = krippendorff.alpha(
            np.array(list(direct_cls_preds.values())),
            level_of_measurement="nominal",
            value_domain=[0, 1],
        )
        sci_alpha = krippendorff.alpha(
            np.array(list(sarc_cls_preds.values())),
            level_of_measurement="nominal",
            value_domain=[0, 1],
        )
        lci_alpha = krippendorff.alpha(
            np.array(list(non_sarc_cls_preds.values())),
            level_of_measurement="nominal",
            value_domain=[0, 1],
        )
        tsc_alpha = krippendorff.alpha(
            np.array(list(neutral_cls_preds.values())),
            level_of_measurement="nominal",
            value_domain=[0, 1, 2],
        )

        model_names.append(model_name)
        results.append(
            [
                bsc_alpha,
                tsc_alpha,
                sci_alpha,
                lci_alpha,
            ]
        )

    return pd.DataFrame(
        results,
        columns=[
            "BSC",
            "TSC",
            "SCS",
            "LCS",
        ],
        index=model_names,
    )


def prompt_variant_rationale_consistency_analysis(
    all_sample_prediction_info: dict[str, list[SampleTaskOutputInfo]],
) -> pd.DataFrame:
    model_names = []
    results = []
    for model_name, sample_prediction_infos in all_sample_prediction_info.items():
        direct_exp_pairs = [[], []]
        neutral_exp_pairs = [[], []]
        sarc_exp_pairs = [[], []]
        non_sarc_exp_pairs = [[], []]

        # BSC
        for sample in sample_prediction_infos:
            pred1s = []
            pred2s = []
            for i, prediction1 in enumerate(sample.bsc_task_model_outputs.values()):
                for j, prediction2 in enumerate(sample.bsc_task_model_outputs.values()):
                    if i == j or i > j:
                        continue
                    if prediction1.prediction_label != prediction2.prediction_label:
                        continue
                    pred1s.append(prediction1.rationale)
                    pred2s.append(prediction2.rationale)
            direct_exp_pairs[0].extend(pred1s)
            direct_exp_pairs[1].extend(pred2s)

        f1s = []
        for i in range(0, len(direct_exp_pairs[0]), BERT_SCORE_PARTITION_SIZE):
            f1 = bert_score(
                direct_exp_pairs[0][i : i + BERT_SCORE_PARTITION_SIZE],
                direct_exp_pairs[1][i : i + BERT_SCORE_PARTITION_SIZE],
                model_name_or_path=BERT_SCORE_MODEL,
                batch_size=BERT_SCORE_BATCH_SIZE,
                device="cuda",
            )["f1"]
            f1s.append(f1)
            gc.collect()
        f1 = torch.cat(f1s)
        bsc_rationale_sim = f1.mean().item()
        bsc_rationale_sim_std = f1.std().item()
        del direct_exp_pairs
        gc.collect()

        # SCI
        for sample in sample_prediction_infos:
            # sarcastic perspective
            pred1s = []
            pred2s = []
            for i, prediction1 in enumerate(sample.scs_task_model_outputs.values()):
                for j, prediction2 in enumerate(sample.scs_task_model_outputs.values()):
                    if i == j or i > j:
                        continue
                    if not prediction1.is_successful or not prediction2.is_successful:
                        continue
                    if prediction1.score > 0.5 and prediction2.score <= 0.5:
                        continue
                    if prediction1.score <= 0.5 and prediction2.score > 0.5:
                        continue
                    pred1s.append(prediction1.rationale)
                    pred2s.append(prediction2.rationale)
            sarc_exp_pairs[0].extend(pred1s)
            sarc_exp_pairs[1].extend(pred2s)

        f1s = []
        for i in range(0, len(sarc_exp_pairs[0]), BERT_SCORE_PARTITION_SIZE):
            f1 = bert_score(
                sarc_exp_pairs[0][i : i + BERT_SCORE_PARTITION_SIZE],
                sarc_exp_pairs[1][i : i + BERT_SCORE_PARTITION_SIZE],
                model_name_or_path=BERT_SCORE_MODEL,
                batch_size=BERT_SCORE_BATCH_SIZE,
                device="cuda",
            )["f1"]
            f1s.append(f1)
            gc.collect()
        f1 = torch.cat(f1s)
        f1 = cast(torch.Tensor, f1)
        sci_rationale_sim = f1.mean().item()
        sci_rationale_sim_std = f1.std().item()
        del sarc_exp_pairs
        gc.collect()

        # LCI
        for sample in sample_prediction_infos:
            # non-sarcastic perspective
            pred1s = []
            pred2s = []
            for i, prediction1 in enumerate(sample.lcs_task_model_outputs.values()):
                for j, prediction2 in enumerate(sample.lcs_task_model_outputs.values()):
                    if i == j or i > j:
                        continue
                    if not prediction1.is_successful or not prediction2.is_successful:
                        continue
                    if prediction1.score >= 0.5 and prediction2.score < 0.5:
                        continue
                    if prediction1.score < 0.5 and prediction2.score >= 0.5:
                        continue
                    pred1s.append(prediction1.rationale)
                    pred2s.append(prediction2.rationale)
            non_sarc_exp_pairs[0].extend(pred1s)
            non_sarc_exp_pairs[1].extend(pred2s)

        f1s = []
        for i in range(0, len(non_sarc_exp_pairs[0]), BERT_SCORE_PARTITION_SIZE):
            f1 = bert_score(
                non_sarc_exp_pairs[0][i : i + BERT_SCORE_PARTITION_SIZE],
                non_sarc_exp_pairs[1][i : i + BERT_SCORE_PARTITION_SIZE],
                model_name_or_path=BERT_SCORE_MODEL,
                batch_size=BERT_SCORE_BATCH_SIZE,
                device="cuda",
            )["f1"]
            f1s.append(f1)
            gc.collect()
        f1 = torch.cat(f1s)
        f1 = cast(torch.Tensor, f1)
        lci_rationale_sim = f1.mean().item()
        lci_rationale_sim_std = f1.std().item()
        del non_sarc_exp_pairs
        gc.collect()

        # TSC
        for sample in sample_prediction_infos:
            # neutral classification
            pred1s = []
            pred2s = []
            for i, prediction1 in enumerate(sample.tsc_task_model_outputs.values()):
                for j, prediction2 in enumerate(sample.tsc_task_model_outputs.values()):
                    if i == j or i > j:
                        continue
                    if prediction1.prediction_label != prediction2.prediction_label:
                        continue
                    pred1s.append(prediction1.rationale)
                    pred2s.append(prediction2.rationale)
            neutral_exp_pairs[0].extend(pred1s)
            neutral_exp_pairs[1].extend(pred2s)

        f1s = []
        for i in range(0, len(neutral_exp_pairs[0]), BERT_SCORE_PARTITION_SIZE):
            f1 = bert_score(
                neutral_exp_pairs[0][i : i + BERT_SCORE_PARTITION_SIZE],
                neutral_exp_pairs[1][i : i + BERT_SCORE_PARTITION_SIZE],
                model_name_or_path=BERT_SCORE_MODEL,
                batch_size=BERT_SCORE_BATCH_SIZE,
                device="cuda",
            )["f1"]
            f1s.append(f1)
            gc.collect()
        f1 = torch.cat(f1s)
        tsc_rationale_sim = f1.mean().item()
        tsc_rationale_sim_std = f1.std().item()
        del neutral_exp_pairs
        gc.collect()

        model_names.append(model_name)
        results.append(
            [
                bsc_rationale_sim,
                bsc_rationale_sim_std,
                tsc_rationale_sim,
                tsc_rationale_sim_std,
                sci_rationale_sim,
                sci_rationale_sim_std,
                lci_rationale_sim,
                lci_rationale_sim_std,
            ]
        )

    return pd.DataFrame(
        results,
        columns=[
            "BSC",
            "BSC_std",
            "TSC",
            "TSC_std",
            "SCS",
            "SCS_std",
            "LCS",
            "LCS_std",
        ],
        index=model_names,
    )


class PromptVariantConsistencyReporter(BaseReporter):
    name = "inter-prompt-consistency"

    def get_results(self) -> dict[str, pd.DataFrame]:
        cls_consistency = prompt_variant_cls_consistency_analysis(
            self.all_sample_task_output_info
        )
        exp_consistency = prompt_variant_rationale_consistency_analysis(
            self.all_sample_task_output_info
        )
        return {
            "cls_consistency": cls_consistency,
            "rationale_consistency": exp_consistency,
        }

    def _draw_cls_consistency(self) -> None:
        if self.results is None:
            return

        cls_consistency_df = self.results["cls_consistency"]

        sorted_model_names = []
        for n in self.sorted_model_names:
            if n in cls_consistency_df.index:
                sorted_model_names.append(n)
        cls_consistency_df = cls_consistency_df.loc[sorted_model_names]
        cls_consistency_df.index = cls_consistency_df.index.map(
            self.get_model_name_alias
        )

        cls_consistency_df["Mean"] = cls_consistency_df.mean(axis=1, skipna=True)
        col_means = cls_consistency_df.drop("Mean", axis=1).mean(axis=0, skipna=True)
        mean_row = pd.concat([col_means, pd.Series([0], index=["Mean"])]).to_frame().T
        mean_row.index = ["Mean"]
        cls_consistency_df = pd.concat([cls_consistency_df, mean_row])

        mask = np.zeros_like(cls_consistency_df, dtype=bool)
        mask[-1, -1] = True  # 最后一行最后一列设为True
        annot = cls_consistency_df.copy().map(
            lambda x: f"{x:.4f}" if not pd.isna(x) else ""
        )
        annot.iloc[-1, -1] = ""
        cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
        cmap.set_bad(color="lightgray")

        sns.set_theme(context="talk", style="white", font="Arial", font_scale=1.2)

        f, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cls_consistency_df,
            cmap=cmap,
            annot=True,
            fmt=".4f",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
            # annot_kws={"size": 14},
            center=0,
            vmin=-1,
            vmax=1,
            mask=mask,
        )
        ax.axhline(
            y=len(cls_consistency_df) - 1,
            color="white",
            linewidth=5,
            xmin=0,
            xmax=1,
        )
        ax.axvline(
            x=len(cls_consistency_df.columns) - 1,
            color="white",
            linewidth=5,
            ymin=0,
            ymax=1,
        )
        ax.xaxis.set_ticks_position("top")
        ax.yaxis.set_ticks_position("left")
        # Set x and y axis labels
        ax.set_xlabel("Task")
        ax.set_ylabel("Model")

        # Increase font size for tick labels
        # ax.tick_params(axis="x", labelsize=14)
        # ax.tick_params(axis="y", labelsize=14)

        # Place x-axis label at the top
        ax.xaxis.set_label_position("top")

        # Adjust colorbar tick label font size
        cbar = ax.collections[0].colorbar
        # cbar.ax.tick_params(labelsize=14)  # Set colorbar tick label size

        # Remove the border line between the heatmap and axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        f.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)

        f.savefig(
            str(self.output_path / "cls_consistency.pdf"), dpi=300, bbox_inches="tight"
        )

    def _draw_rationale_consistency(self) -> None:
        if self.results is None:
            return

        rationale_consistency = self.results["rationale_consistency"]

        sorted_model_names = []
        for n in self.sorted_model_names:
            if n in rationale_consistency.index:
                sorted_model_names.append(n)
        rationale_consistency = rationale_consistency.loc[sorted_model_names]
        rationale_consistency.index = rationale_consistency.index.map(
            self.get_model_name_alias
        )

        # Convert from wide format to long format
        metrics = ["BSC", "TSC", "SCS", "LCS"]
        df = rationale_consistency.reset_index().rename(columns={"index": "Model"})
        df_long = pd.DataFrame()

        for metric in metrics:
            if metric in df.columns and f"{metric}_std" in df.columns:
                temp_df = df[["Model", metric, f"{metric}_std"]].copy()
                temp_df.rename(
                    columns={metric: "Score", f"{metric}_std": "Std"}, inplace=True
                )
                temp_df["Task"] = metric
                df_long = pd.concat([df_long, temp_df], ignore_index=True)

        # Drop rows with NaN in the "Score" column
        df_long = df_long.dropna(subset=["Score"])

        # Set a theme and custom color palette
        sns.set_theme(context="talk", style="white", font="Arial", font_scale=1.8)
        color_palette = sns.color_palette(
            "Set2", len(metrics)
        )  # Using 'Set2' for distinct colors

        plt.figure(figsize=(8, 12))

        # Create a horizontal barplot (Model on the y-axis, Score on the x-axis)
        ax = sns.barplot(
            data=df_long,
            y="Model",
            x="Score",
            hue="Task",
            palette=color_palette,
            orient="h",  # Explicitly define horizontal orientation
            edgecolor="black",  # Add a border to each bar
            linewidth=1.5,  # Set the border width
        )

        # Add horizontal error bars
        # Each bar is returned as a patch in ax.patches.
        # We draw the error bar at the midpoint of the bar on the y-axis (bar center),
        # and extend it horizontally (xerr) by the standard deviation.
        for i, bar in enumerate(ax.patches):
            if i >= len(df_long):
                break
            # Calculate the center of the bar on the y-axis
            y_center = bar.get_y() + bar.get_height() / 2

            # Current bar's Score is its width
            # We use 'Std' from the df_long to determine the error range
            std = df_long["Std"].iloc[i]
            score = df_long["Score"].iloc[i]

            # Plot the error bar horizontally
            ax.errorbar(
                score,  # x-position (bar's right end is the score)
                y_center,  # y-position (vertical center of the bar)
                xerr=std,  # horizontal error
                color="black",
                capsize=3,
                lw=1.5,
                elinewidth=1.5,
                capthick=1.5,
                fmt="none",  # no marker
            )

        # Rotate x-axis labels if needed (usually horizontal is fine for scores)
        plt.xticks(rotation=0)
        plt.xlabel("Score")
        plt.ylabel("Model")

        # Move the legend above the plot
        # plt.legend(title="", loc="best")

        plt.savefig(
            str(self.output_path / "rationale_consistency.pdf"),
            dpi=300,
            bbox_inches="tight",
        )

    def draw(self) -> None:
        self._draw_cls_consistency()
        self._draw_rationale_consistency()
