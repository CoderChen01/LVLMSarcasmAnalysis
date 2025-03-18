from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchmetrics.functional.classification import stat_scores

from .base import BaseReporter, SampleTaskOutputInfo


def ground_truth_agreement_analysis(
    all_sample_prediction_info: dict[str, list[SampleTaskOutputInfo]],
) -> dict[str, pd.DataFrame]:

    model_names = []

    bsc_results = []
    scs_results = []
    lcs_results = []
    scs_lcs_results = []
    tsc_results = []

    all_bsc_preds = []
    all_bsc_human_labels = []
    all_bsc_correct_idx = []

    all_scs_lcs_preds = []
    all_scs_lcs_human_labels = []
    all_scs_lcs_correct_idx = []

    all_scs_preds = []
    all_scs_human_labels = []
    all_scs_correct_idx = []

    all_lcs_preds = []
    all_lcs_human_labels = []
    all_lcs_correct_idx = []

    all_tsc_preds = []
    all_tsc_human_labels = []
    all_tsc_correct_idx = []

    for model_name, sample_prediction_infos in all_sample_prediction_info.items():

        bsc_preds = []
        bsc_human_labels = []
        bsc_correct_idx = []
        for idx, sample in enumerate(sample_prediction_infos):
            # direct classification
            counter = [0, 0]
            for prediction in sample.bsc_task_model_outputs.values():
                counter[prediction.prediction_label] += 1
            if counter[0] > counter[1]:
                bsc_preds.append(0)
            elif counter[0] < counter[1]:
                bsc_preds.append(1)
            else:
                continue
            bsc_human_labels.append(sample.human_label)
            bsc_correct_idx.append(idx)

        tsc_preds = []
        tsc_human_labels = []
        tsc_correct_idx = []
        for idx, sample in enumerate(sample_prediction_infos):
            # neutral perspective
            counter = [0, 0, 0]
            for prediction in sample.tsc_task_model_outputs.values():
                counter[prediction.prediction_label] += 1
            if counter[0] > counter[1] and counter[0] > counter[2]:
                tsc_preds.append(0)
            elif counter[1] > counter[2] and counter[1] > counter[0]:
                tsc_preds.append(1)
            else:
                continue
            tsc_human_labels.append(sample.human_label)
            tsc_correct_idx.append(idx)

        scs_preds = []
        scs_human_labels = []
        scs_correct_idx = []
        for idx, sample in enumerate(sample_prediction_infos):
            # sarcastic perspective
            counter = [0, 0]
            for prediction in sample.scs_task_model_outputs.values():
                if not prediction.is_successful:
                    continue
                if prediction.score > 0.5:
                    counter[1] += 1
                else:
                    counter[0] += 1
            if counter[0] > counter[1]:
                scs_preds.append(0)
            elif counter[1] > counter[0]:
                scs_preds.append(1)
            else:
                continue
            scs_human_labels.append(sample.human_label)
            scs_correct_idx.append(idx)

        lcs_preds = []
        lcs_human_labels = []
        lcs_correct_idx = []
        for idx, sample in enumerate(sample_prediction_infos):
            # sarcastic perspective
            counter = [0, 0]
            for prediction in sample.lcs_task_model_outputs.values():
                if not prediction.is_successful:
                    continue
                if prediction.score < 0.5:
                    counter[1] += 1
                else:
                    counter[0] += 1
            if counter[0] > counter[1]:
                lcs_preds.append(0)
            elif counter[1] > counter[0]:
                lcs_preds.append(1)
            else:
                continue
            lcs_human_labels.append(sample.human_label)
            lcs_correct_idx.append(idx)

        scs_lcs_preds = []
        scs_lcs_human_labels = []
        scs_lcs_correct_idx = []
        for idx, sample in enumerate(sample_prediction_infos):
            # comparison
            counter = [0, 0]
            for sarc_prediction in sample.scs_task_model_outputs.values():
                for non_sarc_prediction in sample.lcs_task_model_outputs.values():
                    if (
                        sarc_prediction.is_successful
                        and non_sarc_prediction.is_successful
                    ):
                        if sarc_prediction.score > non_sarc_prediction.score:
                            counter[1] += 1
                        else:
                            counter[0] += 1
                    else:
                        continue
            if counter[0] > counter[1]:
                scs_lcs_preds.append(0)
            elif counter[0] < counter[1]:
                scs_lcs_preds.append(1)
            else:
                continue
            scs_lcs_human_labels.append(sample.human_label)
            scs_lcs_correct_idx.append(idx)

        model_names.append(model_name)

        all_bsc_human_labels.append(bsc_human_labels)
        all_lcs_human_labels.append(lcs_human_labels)
        all_scs_human_labels.append(scs_human_labels)
        all_scs_lcs_human_labels.append(scs_lcs_human_labels)
        all_tsc_human_labels.append(tsc_human_labels)

        all_bsc_preds.append(bsc_preds)
        all_tsc_preds.append(tsc_preds)
        all_scs_lcs_preds.append(scs_lcs_preds)
        all_scs_preds.append(scs_preds)
        all_lcs_preds.append(lcs_preds)

        all_bsc_correct_idx.append(bsc_correct_idx)
        all_scs_lcs_correct_idx.append(scs_lcs_correct_idx)
        all_scs_correct_idx.append(scs_correct_idx)
        all_lcs_correct_idx.append(lcs_correct_idx)
        all_tsc_correct_idx.append(tsc_correct_idx)

        # direct classification
        tp, fp, tn, fn, _ = stat_scores(
            torch.tensor(bsc_preds),
            torch.tensor(bsc_human_labels),
            task="binary",
        ).tolist()
        bsc_results.append(
            (
                np.round(
                    np.array([(tp + tn), tp, fp, tn, fn]) / (tp + tn + fp + fn) * 100, 2
                )
            ).tolist()
        )

        # sarc
        tp, fp, tn, fn, _ = stat_scores(
            torch.tensor(scs_preds),
            torch.tensor(scs_human_labels),
            task="binary",
        ).tolist()
        scs_results.append(
            (
                np.round(
                    np.array([(tp + tn), tp, fp, tn, fn]) / (tp + tn + fp + fn) * 100, 2
                )
            ).tolist()
        )

        # lcs
        tp, fp, tn, fn, _ = stat_scores(
            torch.tensor(lcs_preds),
            torch.tensor(lcs_human_labels),
            task="binary",
        ).tolist()
        lcs_results.append(
            (
                np.round(
                    np.array([(tp + tn), tp, fp, tn, fn]) / (tp + tn + fp + fn) * 100, 2
                )
            ).tolist()
        )

        # comparison
        tp, fp, tn, fn, _ = stat_scores(
            torch.tensor(scs_lcs_preds),
            torch.tensor(scs_lcs_human_labels),
            task="binary",
        ).tolist()
        scs_lcs_results.append(
            (
                np.round(
                    np.array([(tp + tn), tp, fp, tn, fn]) / (tp + tn + fp + fn) * 100, 2
                )
            ).tolist()
        )

        # neutral
        tp, fp, tn, fn, _ = stat_scores(
            torch.tensor(tsc_preds),
            torch.tensor(tsc_human_labels),
            task="binary",
        ).tolist()
        tsc_results.append(
            (
                np.round(
                    np.array([(tp + tn), tp, fp, tn, fn]) / (tp + tn + fp + fn) * 100, 2
                )
            ).tolist()
        )

    bsc_cls_result_df = pd.DataFrame(
        bsc_results,
        columns=["acc", "tp", "fp", "tn", "fn"],
        index=model_names,
    )
    scs_cls_result_df = pd.DataFrame(
        scs_results,
        columns=["acc", "tp", "fp", "tn", "fn"],
        index=model_names,
    )
    lcs_cls_result_df = pd.DataFrame(
        lcs_results,
        columns=["acc", "tp", "fp", "tn", "fn"],
        index=model_names,
    )
    comp_cls_result_df = pd.DataFrame(
        scs_lcs_results,
        columns=["acc", "tp", "fp", "tn", "fn"],
        index=model_names,
    )
    tsc_cls_result_df = pd.DataFrame(
        tsc_results,
        columns=["acc", "tp", "fp", "tn", "fn"],
        index=model_names,
    )
    num_all_samples = len(sample_prediction_infos)
    correctness_result_df = pd.DataFrame(
        np.asarray(
            [
                [
                    round((len(human_labels) / num_all_samples) * 100, 2)
                    for human_labels in all_bsc_human_labels
                ],
                [
                    round((len(human_labels) / num_all_samples) * 100, 2)
                    for human_labels in all_tsc_human_labels
                ],
                [
                    round((len(human_labels) / num_all_samples) * 100, 2)
                    for human_labels in all_scs_human_labels
                ],
                [
                    round((len(human_labels) / num_all_samples) * 100, 2)
                    for human_labels in all_lcs_human_labels
                ],
                [
                    round((len(human_labels) / num_all_samples) * 100, 2)
                    for human_labels in all_scs_lcs_human_labels
                ],
            ]
        ).T.tolist(),
        columns=[
            "BSC",
            "TSC",
            "SCS",
            "LCS",
            "COMP",
        ],
        index=model_names,
    )

    all_voting_results = {}
    for task, all_preds, all_human_labels, all_correct_idx in zip(
        ["BSC", "TSC", "SCS", "LCS", "COMP"],
        [
            all_bsc_preds,
            all_tsc_preds,
            all_scs_preds,
            all_lcs_preds,
            all_scs_lcs_preds,
        ],
        [
            all_bsc_human_labels,
            all_tsc_human_labels,
            all_scs_human_labels,
            all_lcs_human_labels,
            all_scs_lcs_human_labels,
        ],
        [
            all_bsc_correct_idx,
            all_tsc_correct_idx,
            all_scs_correct_idx,
            all_lcs_correct_idx,
            all_scs_lcs_correct_idx,
        ],
    ):
        all_preds_intersection = []
        human_labels_intersection = []
        correct_idx_intersection = set.intersection(*map(set, all_correct_idx))
        for preds, human_labels, correct_idx in zip(
            all_preds, all_human_labels, all_correct_idx
        ):
            preds_intersection = []
            human_labels_intersection = []
            for idx in correct_idx_intersection:
                i = correct_idx.index(idx)
                preds_intersection.append(preds[i])
                human_labels_intersection.append(human_labels[i])
            all_preds_intersection.append(preds_intersection)

        voting_preds = []
        voting_human_labels = []
        for i, preds in enumerate(zip(*all_preds_intersection)):
            counter = [0, 0]
            for pred in preds:
                counter[pred] += 1
            if counter[0] > counter[1]:
                voting_preds.append(0)
                voting_human_labels.append(human_labels_intersection[i])
            elif counter[0] < counter[1]:
                voting_preds.append(1)
                voting_human_labels.append(human_labels_intersection[i])
            else:
                continue

        tp, fp, tn, fn, _ = stat_scores(
            torch.tensor(voting_preds),
            torch.tensor(voting_human_labels),
            task="binary",
        ).tolist()
        all_voting_results[task] = (
            np.round(
                np.array([(tp + tn), tp, fp, tn, fn]) / (tp + tn + fp + fn) * 100,
                2,
            )
        ).tolist() + [len(voting_preds) / num_all_samples * 100]

    return {
        "BSC": bsc_cls_result_df,
        "TSC": tsc_cls_result_df,
        "SCS": scs_cls_result_df,
        "LCS": lcs_cls_result_df,
        "COMP": comp_cls_result_df,
        "CORRECTNESS": correctness_result_df,
        "VOITING": pd.DataFrame(
            all_voting_results,
            index=["acc", "tp", "fp", "tn", "fn", "correctness"],
        ),
    }


class GroundTruthAgreementReporter(BaseReporter):
    name = "ground-truth-agreement"

    def get_results(self) -> dict[str, pd.DataFrame]:
        return ground_truth_agreement_analysis(self.all_sample_task_output_info)

    def _sort_model_names(
        self, df: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        sorted_model_names = []
        for n in self.sorted_model_names:
            if n in df.index:
                sorted_model_names.append(n)
        df = df.loc[sorted_model_names]
        df.index = df.index.map(self.get_model_name_alias)
        return df

    def _draw_accuracy_heatmap(self) -> None:
        if self.results is None:
            return

        bsc_accuracy = self._sort_model_names(
            self.results["BSC"]["acc"] * self.results["CORRECTNESS"]["BSC"] / 100
        )
        tsc_accuracy = self._sort_model_names(
            self.results["TSC"]["acc"] * self.results["CORRECTNESS"]["TSC"] / 100
        )
        scs_accuracy = self._sort_model_names(
            self.results["SCS"]["acc"] * self.results["CORRECTNESS"]["SCS"] / 100
        )
        lcs_accuracy = self._sort_model_names(
            self.results["LCS"]["acc"] * self.results["CORRECTNESS"]["LCS"] / 100
        )
        comp_accuracy = self._sort_model_names(
            self.results["COMP"]["acc"] * self.results["CORRECTNESS"]["COMP"] / 100
        )
        acc_df = pd.DataFrame(
            {
                "BSC": bsc_accuracy,
                "TSC": tsc_accuracy,
                "SCS": scs_accuracy,
                "LCS": lcs_accuracy,
                "COMP": comp_accuracy,
            },
            index=bsc_accuracy.index,
        )

        acc_df["Mean"] = acc_df.mean(axis=1, skipna=True)
        col_means = acc_df.drop("Mean", axis=1).mean(axis=0, skipna=True)
        mean_row = pd.concat([col_means, pd.Series([0], index=["Mean"])]).to_frame().T
        mean_row.index = ["Mean"]
        acc_df = pd.concat([acc_df, mean_row])

        mask = np.zeros_like(acc_df, dtype=bool)
        mask[-1, -1] = True  # 最后一行最后一列设为True
        annot = acc_df.copy().map(lambda x: f"{x:.2f}" if not pd.isna(x) else "")
        annot.iloc[-1, -1] = ""
        cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
        cmap.set_bad(color="white")

        sns.set_theme(context="talk", style="white", font="Arial", font_scale=1.2)

        f, ax = plt.subplots(figsize=(6, 6))

        sns.heatmap(
            acc_df,
            cmap=cmap,
            annot=annot,
            fmt="",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 14},
            center=0,
            mask=mask,
        )
        ax.axhline(
            y=len(acc_df) - 1,
            color="white",
            linewidth=5,
            xmin=0,
            xmax=1,
        )
        ax.axvline(
            x=len(acc_df.columns) - 1,
            color="white",
            linewidth=5,
            ymin=0,
            ymax=1,
        )
        ax.xaxis.set_ticks_position("top")
        ax.yaxis.set_ticks_position("left")
        # Set x and y axis labels
        ax.set_xlabel("Task", fontsize=16)
        ax.set_ylabel("Model", fontsize=16)

        # Increase font size for tick labels
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

        # Place x-axis label at the top
        ax.xaxis.set_label_position("top")

        # Adjust colorbar tick label font size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)  # Set colorbar tick label size

        # Remove the border line between the heatmap and axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        f.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)

        f.savefig(
            str(self.output_path / "ground_truth_agreement_accuracy.pdf"),
            dpi=300,
            bbox_inches="tight",
        )

    def _draw_barplot(self) -> None:
        if self.results is None:
            return

        bsc_df = self._sort_model_names(self.results["BSC"].iloc[:, 1:])
        bsc_correctness_df = self._sort_model_names(self.results["CORRECTNESS"]["BSC"])

        tsc_df = self._sort_model_names(self.results["TSC"].iloc[:, 1:])
        tsc_correctness_df = self._sort_model_names(self.results["CORRECTNESS"]["TSC"])

        scs_df = self._sort_model_names(self.results["SCS"].iloc[:, 1:])
        scs_correctness_df = self._sort_model_names(self.results["CORRECTNESS"]["SCS"])

        lcs_df = self._sort_model_names(self.results["LCS"].iloc[:, 1:])
        lcs_correctness_df = self._sort_model_names(self.results["CORRECTNESS"]["LCS"])

        comp_df = self._sort_model_names(self.results["COMP"].iloc[:, 1:])
        comp_correctness_df = self._sort_model_names(
            self.results["CORRECTNESS"]["COMP"]
        )

        sns.set_theme(context="talk", style="white", font="Arial", font_scale=1.2)
        fig, axes = plt.subplots(1, 5, figsize=(20, 6))  # 1 row and 5 columns

        for i, (df, correctness_df, task) in enumerate(
            zip(
                [bsc_df, tsc_df, scs_df, lcs_df, comp_df],
                [
                    bsc_correctness_df,
                    tsc_correctness_df,
                    scs_correctness_df,
                    lcs_correctness_df,
                    comp_correctness_df,
                ],
                ["BSC", "TSC", "SCS", "LCS", "COMP"],
            )
        ):
            df["TP"] = df["tp"] * (correctness_df / 100)
            df["FP"] = df["fp"] * (correctness_df / 100)
            df["TN"] = df["tn"] * (correctness_df / 100)
            df["FN"] = df["fn"] * (correctness_df / 100)

            df = df.iloc[::-1]
            ax = axes[i]
            ax.barh(df.index, df["TP"], color="green", label="TP", edgecolor="black")
            ax.barh(
                df.index,
                df["FP"],
                left=df["TP"],
                color="red",
                label="FP",
                edgecolor="black",
            )
            ax.barh(
                df.index,
                df["TN"],
                left=df["TP"] + df["FP"],
                color="blue",
                label="TN",
                edgecolor="black",
            )
            ax.barh(
                df.index,
                df["FN"],
                left=df["TP"] + df["FP"] + df["TN"],
                color="orange",
                label="FN",
                edgecolor="black",
            )

            ax.set_xlabel("Correctness (%)")
            ax.set_title(task)
            if i == 0:
                ax.set_ylabel("Model")
                # ax.set_yticklabels(df.index, fontsize=12)
            else:
                # Remove y-axis labels and ticks for all subplots except the first one
                ax.set_yticks([])
                ax.set_yticklabels([])

            # No per-plot legends, instead a single legend on the far right
            # ax.legend_.remove()  # Remove per-plot legends

        # Create a single legend on the far right side
        fig.legend(
            loc="center right",
            bbox_to_anchor=(1.06, 0.5),
            labels=["TP", "FP", "TN", "FN"],
            title="Metric",
        )

        # Adjust layout and save the figure
        plt.tight_layout()

        plt.savefig(
            str(self.output_path / "ground_truth_agreement_stacked_bar.svg"),
            dpi=300,
            bbox_inches="tight",
        )

    def _draw_voting_results(self) -> None:
        if self.results is None:
            return

        voting_results = self.results["VOITING"].T

        voting_results["TP"] = voting_results["tp"] * (
            voting_results["correctness"] / 100
        )
        voting_results["FP"] = voting_results["fp"] * (
            voting_results["correctness"] / 100
        )
        voting_results["TN"] = voting_results["tn"] * (
            voting_results["correctness"] / 100
        )
        voting_results["FN"] = voting_results["fn"] * (
            voting_results["correctness"] / 100
        )
        df = voting_results.iloc[::-1]

        sns.set_theme(context="talk", style="white", font="Arial", font_scale=1.2)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.barh(df.index, df["TP"], color="green", label="TP", edgecolor="black")
        ax.barh(
            df.index,
            df["FP"],
            left=df["TP"],
            color="red",
            label="FP",
            edgecolor="black",
        )
        ax.barh(
            df.index,
            df["TN"],
            left=df["TP"] + df["FP"],
            color="blue",
            label="TN",
            edgecolor="black",
        )
        ax.barh(
            df.index,
            df["FN"],
            left=df["TP"] + df["FP"] + df["TN"],
            color="orange",
            label="FN",
            edgecolor="black",
        )

        ax.set_xlabel("Correctness (%)")
        ax.set_ylabel("Task")

        ax.legend(
            labels=["TP", "FP", "TN", "FN"],
            title="Metric",
        )

        plt.tight_layout()
        plt.savefig(
            str(self.output_path / "ground_truth_agreement_voting_stacked_bar.pdf"),
            dpi=300,
            bbox_inches="tight",
        )

    def draw(self) -> None:
        self._draw_accuracy_heatmap()
        self._draw_barplot()
        self._draw_voting_results()
