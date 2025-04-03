from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torchmetrics.functional.text import bert_score

from .base import BaseReporter, SampleTaskOutputInfo


def neutral_sample_overlap_rate(
    false_sample_ids: set, neutral_sample_ids: set
) -> float:
    intersection = false_sample_ids.intersection(neutral_sample_ids)
    # union = false_sample_ids.union(neutral_sample_ids)
    if len(neutral_sample_ids) == 0:
        return 0.0
    return len(intersection) / len(neutral_sample_ids)


def jacard_similarity_rate(set_a: set, set_b: set) -> float:
    intersection = set_a.intersection(set_b)
    min_size = min(len(set_a), len(set_b))
    if min_size == 0:
        return 0.0
    return len(intersection) / min_size


def calc_neutral_overlap_across_gt(
    sarcastic_gts: list[str],
    non_sarcastic_gts: list[str],
    comp_neutral_sample_ids: list[str],
    tsc_neutral_sample_ids: list[str],
) -> list[float]:
    comp_set = set(comp_neutral_sample_ids)
    tsc_set = set(tsc_neutral_sample_ids)

    sarc_gt_set = set(sarcastic_gts)
    non_sarc_gt_set = set(non_sarcastic_gts)

    sarc_comp = neutral_sample_overlap_rate(sarc_gt_set, comp_set)
    non_sarc_comp = neutral_sample_overlap_rate(non_sarc_gt_set, comp_set)

    sarc_tsc = neutral_sample_overlap_rate(sarc_gt_set, tsc_set)
    non_sarc_tsc = neutral_sample_overlap_rate(non_sarc_gt_set, tsc_set)

    return [
        sarc_comp,
        non_sarc_comp,
        sarc_tsc,
        non_sarc_tsc,
    ]


def neutral_sample_overlap_analysis(
    all_sample_prediction_info: dict[str, list[SampleTaskOutputInfo]],
) -> dict[str, pd.DataFrame]:

    model_names = []

    all_sarc_cls_preds = []
    all_non_sarc_cls_preds = []
    all_comp_neutral_sample_ids = []
    all_tsc_neutral_sample_ids = []

    sim_results = []
    overlap_across_gts = []

    sarcastic_gts = []
    non_sarcastic_gts = []

    for model_name, sample_prediction_infos in all_sample_prediction_info.items():

        if len(sarcastic_gts) == 0 and len(non_sarcastic_gts) == 0:
            for idx, sample in enumerate(sample_prediction_infos):
                if sample.human_label == 1:
                    sarcastic_gts.append(sample.id)
                else:
                    non_sarcastic_gts.append(sample.id)

        sarc_cls_preds = []
        non_sarc_cls_preds = []
        comp_neutral_sample_ids = []
        tsc_neutral_sample_ids = []
        for idx, sample in enumerate(sample_prediction_infos):
            # sarcastic perspective
            counter = [0, 0]
            for prediction in sample.scs_task_model_outputs.values():
                if prediction.is_successful:
                    if prediction.score > 0.5:
                        counter[1] += 1
                    else:
                        counter[0] += 1
                else:
                    continue
            if counter[0] > counter[1]:
                sarc_cls_preds.append(0)
            elif counter[0] < counter[1]:
                sarc_cls_preds.append(1)
            else:
                continue

            # non-sarcastic perspective
            counter = [0, 0]
            for prediction in sample.lcs_task_model_outputs.values():
                if prediction.is_successful:
                    if prediction.score < 0.5:
                        counter[1] += 1
                    else:
                        counter[0] += 1
                else:
                    continue
            if counter[0] > counter[1]:
                non_sarc_cls_preds.append(0)
            elif counter[0] < counter[1]:
                non_sarc_cls_preds.append(1)
            else:
                sarc_cls_preds.pop()
                continue

            # neutral sample
            if sarc_cls_preds[-1] != non_sarc_cls_preds[-1]:
                comp_neutral_sample_ids.append(sample.id)

            counter = [0, 0]
            for p in sample.tsc_task_model_outputs.values():
                if p.is_successful:
                    if p.prediction_label == 2:
                        counter[1] += 1
                    else:
                        counter[0] += 1
            if counter[1] > counter[0]:
                tsc_neutral_sample_ids.append(sample.id)

        model_names.append(model_name)
        all_comp_neutral_sample_ids.append(comp_neutral_sample_ids)
        all_tsc_neutral_sample_ids.append(tsc_neutral_sample_ids)

        sim_results.append(
            jacard_similarity_rate(
                set(comp_neutral_sample_ids), set(tsc_neutral_sample_ids)
            )
        )
        overlap_across_gts.append(
            calc_neutral_overlap_across_gt(
                sarcastic_gts,
                non_sarcastic_gts,
                comp_neutral_sample_ids,
                tsc_neutral_sample_ids,
            )
        )

    comp_neutral_sample_ids_intersection = set.intersection(
        *map(set, all_comp_neutral_sample_ids)
    )
    tsc_neutral_sample_ids_intersection = set.intersection(
        *map(set, all_tsc_neutral_sample_ids)
    )

    all_sim_result = jacard_similarity_rate(
        comp_neutral_sample_ids_intersection,
        tsc_neutral_sample_ids_intersection,
    )

    overlap_across_mth_data = [
        [len(comp), len(tsc), sim]
        for comp, tsc, sim in zip(
            all_comp_neutral_sample_ids, all_tsc_neutral_sample_ids, sim_results
        )
    ]
    overlap_across_mth_data.append(
        [
            len(comp_neutral_sample_ids_intersection),
            len(tsc_neutral_sample_ids_intersection),
            all_sim_result,
        ]
    )

    overlap_across_mth_df = pd.DataFrame(
        overlap_across_mth_data,
        index=model_names + ["All"],
        columns=["num_comp", "num_tsc", "sim"],
    )
    overlap_across_gt_df = pd.DataFrame(
        overlap_across_gts,
        index=model_names,
        columns=[
            "sarc_comp",
            "non_sarc_comp",
            "sarc_tsc",
            "non_sarc_tsc",
        ],
    )
    comp_neutral_ids_df = pd.DataFrame(
        [[i] for i in all_comp_neutral_sample_ids],
        index=model_names,
        columns=["neutral_ids"],
    )
    tsc_neutral_ids_df = pd.DataFrame(
        [[i] for i in all_tsc_neutral_sample_ids],
        index=model_names,
        columns=["neutral_ids"],
    )

    return {
        "overlap_across_method": overlap_across_mth_df,
        "overlap_across_gt": overlap_across_gt_df,
        "comp_neutral_ids": comp_neutral_ids_df,
        "tsc_neutral_ids": tsc_neutral_ids_df,
    }


class NeutralLabelOverlapReporter(BaseReporter):
    name = "neutral-label-overlap"

    def get_results(self) -> dict[str, pd.DataFrame]:
        return neutral_sample_overlap_analysis(self.all_sample_task_output_info)

    def _draw_overlap_across_method(self) -> None:
        if self.results is None:
            return

        overlap_across_mtd_df = self.results["overlap_across_method"]
        overlap_across_mtd_df.rename(
            columns={"num_comp": "SCS-LCS", "num_tsc": "TSC", "sim": "Simularity"},
            inplace=True,
        )

        sns.set_theme(context="paper", style="white", font="Arial", font_scale=1.8)

        overlap_across_mtd_df["Simularity"] = (
            overlap_across_mtd_df["Simularity"] * 100
        ).round(
            2
        )  # Convert similarity to percentage
        overlap_across_mtd_df = pd.DataFrame(
            self._sort_model_names(overlap_across_mtd_df)
        )

        # Split data for separate plots
        df_comp_tsc = (
            overlap_across_mtd_df.iloc[:, :2]
            .reset_index()
            .melt(id_vars="index", var_name="Type", value_name="Count")
        )
        df_sim = (
            overlap_across_mtd_df.iloc[:, [2]]
            .reset_index()
            .melt(id_vars="index", var_name="Type", value_name="Overlap (%)")
        )

        # Set theme for academic-style plot
        sns.set_theme(context="paper", style="white", font="Arial", font_scale=1.8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        bright_palette = ["#E63946", "#F4A261", "#2A9D8F"]

        # Plot COMP and TSC using seaborn
        sns.barplot(
            data=df_comp_tsc,
            y="index",
            x="Count",
            hue="Type",
            ax=axes[0],
            palette=bright_palette[:2],
            edgecolor="black",
        )
        axes[0].set_xlabel("Count")
        axes[0].set_ylabel("Model")
        axes[0].set_title("Comparison of SCS-LCS and TSC")
        axes[0].legend(title="Type", loc="upper right")
        axes[0].grid(axis="x", linestyle="--", alpha=0.6)

        # Plot Simularity using seaborn with hatch pattern
        sns.barplot(
            data=df_sim,
            y="index",
            x="Overlap (%)",
            hue="Type",
            ax=axes[1],
            palette=[bright_palette[2]],
            edgecolor="black",
        )
        axes[1].set_xlabel("Overlap (%)")
        axes[1].set_ylabel("")  # Remove ylabel for second plot
        axes[1].set_title("Overlap Percentage")
        axes[1].legend_.remove()  # Remove legend from second plot
        axes[1].grid(axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        fig.savefig(
            str(self.output_path / "overlap_across_method.pdf"),
            dpi=300,
            bbox_inches="tight",
        )

    def _draw_overlap_across_gt(self) -> None:
        if self.results is None:
            return

        overlap_across_gt_df = self.results["overlap_across_gt"]
        comp_df = overlap_across_gt_df.iloc[:, :2].copy()
        comp_df.rename(
            columns={"sarc_comp": "Sarcastic", "non_sarc_comp": "Non-Sarcastic"},
            inplace=True,
        )
        comp_df = self._sort_model_names(comp_df)

        tsc_df = overlap_across_gt_df.iloc[:, 2:].copy()
        tsc_df.rename(
            columns={"sarc_tsc": "Sarcastic", "non_sarc_tsc": "Non-Sarcastic"},
            inplace=True,
        )
        tsc_df = self._sort_model_names(tsc_df)

        sns.set_theme(context="paper", style="white", font="Arial", font_scale=1.8)
        f, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        for i, (data_df, name) in enumerate(zip((comp_df, tsc_df), ("SCS-LCS", "TSC"))):
            ax = axes[i]

            # 转换相似度为百分比，保留两位小数
            data_df = data_df * 100
            data_df = data_df.round(2)

            # 纵坐标为index，横坐标为百分比
            data_df = data_df.reset_index().melt(
                id_vars="index", var_name="Type", value_name="Overlap (%)"
            )

            sns.barplot(
                data=data_df,
                y="index",
                x="Overlap (%)",
                hue="Type",
                ax=ax,
                palette=["#FF4C4C", "#4C9EFF"],  # 颜色符合学术风格
                edgecolor="black",
            )

            ax.set_xlabel("Overlap (%)")
            ax.set_ylabel("Model" if i == 0 else "")
            ax.set_title(name)
            ax.legend(title="Type", loc="best")
            legend = ax.legend(title="Type", loc="upper right", framealpha=0.8)
            legend.get_frame().set_linewidth(0.8)  # 增加边框清晰度

        plt.tight_layout()
        f.savefig(
            str(self.output_path / "overlap_across_gt.pdf"),
            dpi=300,
            bbox_inches="tight",
        )

    def draw(self) -> None:
        self._draw_overlap_across_method()
        self._draw_overlap_across_gt()
