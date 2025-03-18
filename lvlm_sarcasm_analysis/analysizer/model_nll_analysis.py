from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torchmetrics.functional.text import bert_score

from .base import BaseReporter, SampleTaskOutputInfo


def model_nll_analysis(
    all_sample_prediction_info: dict[str, list[SampleTaskOutputInfo]],
) -> pd.DataFrame:
    model_names = []
    results = []
    for model_name, sample_prediction_infos in all_sample_prediction_info.items():
        p_direct_cls_nlls = []
        n_direct_cls_nlls = []

        sarc_tsc_nlls = []
        non_sarc_tsc_nlls = []
        neutral_tsc_nlls = []

        p_sarc_cls_nlls = []
        n_sarc_cls_nlls = []

        p_non_sarc_cls_nlls = []
        n_non_sarc_cls_nlls = []

        for sample in sample_prediction_infos:
            # direct classification
            for prediction in sample.bsc_task_model_outputs.values():
                if prediction.prediction_label == 0:
                    n_direct_cls_nlls.append(prediction.nll)
                elif prediction.prediction_label == 1:
                    p_direct_cls_nlls.append(prediction.nll)

            # TSC
            for prediction in sample.tsc_task_model_outputs.values():
                nll = prediction.nll
                if prediction.prediction_label == 0:
                    non_sarc_tsc_nlls.append(nll)
                elif prediction.prediction_label == 1:
                    sarc_tsc_nlls.append(nll)
                elif prediction.prediction_label == 2:
                    neutral_tsc_nlls.append(nll)

            # sarcastic perspective
            for prediction in sample.scs_task_model_outputs.values():
                nll = prediction.nll
                if prediction.is_successful:
                    if prediction.score > 0.5:
                        p_sarc_cls_nlls.append(nll)
                    else:
                        n_sarc_cls_nlls.append(nll)

            # non-sarcastic perspective
            for prediction in sample.lcs_task_model_outputs.values():
                nll = prediction.nll
                if prediction.is_successful:
                    if prediction.score < 0.5:
                        p_non_sarc_cls_nlls.append(nll)
                    else:
                        n_non_sarc_cls_nlls.append(nll)

        model_names.append(model_name)
        results.append(
            [
                np.mean(p_direct_cls_nlls),
                np.std(p_direct_cls_nlls),
                np.mean(n_direct_cls_nlls),
                np.std(n_direct_cls_nlls),
                np.mean(p_direct_cls_nlls + n_direct_cls_nlls),
                np.std(p_direct_cls_nlls + n_direct_cls_nlls),
                np.mean(p_sarc_cls_nlls),
                np.std(p_sarc_cls_nlls),
                np.mean(n_sarc_cls_nlls),
                np.std(n_sarc_cls_nlls),
                np.mean(p_sarc_cls_nlls + n_sarc_cls_nlls),
                np.std(p_sarc_cls_nlls + n_sarc_cls_nlls),
                np.mean(p_non_sarc_cls_nlls),
                np.std(p_non_sarc_cls_nlls),
                np.mean(n_non_sarc_cls_nlls),
                np.std(n_non_sarc_cls_nlls),
                np.mean(p_non_sarc_cls_nlls + n_non_sarc_cls_nlls),
                np.std(p_non_sarc_cls_nlls + n_non_sarc_cls_nlls),
                np.mean(sarc_tsc_nlls),
                np.std(sarc_tsc_nlls),
                np.mean(non_sarc_tsc_nlls),
                np.std(non_sarc_tsc_nlls),
                np.mean(neutral_tsc_nlls),
                np.std(neutral_tsc_nlls),
                np.mean(sarc_tsc_nlls + non_sarc_tsc_nlls + neutral_tsc_nlls),
                np.std(sarc_tsc_nlls + non_sarc_tsc_nlls + neutral_tsc_nlls),
            ]
        )

    return pd.DataFrame(
        results,
        columns=[
            "p_bsc",
            "p_bsc_std",
            "n_bsc",
            "n_bsc_std",
            "bsc",
            "bsc_std",
            "p_scs",
            "p_scs_std",
            "n_scs",
            "n_scs_std",
            "scs",
            "scs_std",
            "p_lcs",
            "p_lcs_std",
            "n_lcs",
            "n_lcs_std",
            "lcs",
            "lcs_std",
            "sarc_tsc",
            "sarc_tsc_std",
            "non_sarc_tsc",
            "non_sarc_tsc_std",
            "neutral_tsc",
            "neutral_tsc_std",
            "tsc",
            "tsc_std",
        ],
        index=model_names,
    )


class ModelNllReporter(BaseReporter):
    name = "model_nll_analysis"

    def get_results(self) -> dict[str, pd.DataFrame]:
        return {
            "model_nll_analysis": model_nll_analysis(self.all_sample_task_output_info)
        }

    def draw(self) -> None:
        if self.results is None:
            return

        data_df = self.results["model_nll_analysis"]

        bsc_data_df = self._sort_model_names(data_df.iloc[:, :6]).rename(
            columns={
                "p_bsc": "Sarcastic",
                "p_bsc_std": "Sarcastic_std",
                "n_bsc": "Non-Sarcastic",
                "n_bsc_std": "Non-Sarcastic_std",
                "bsc": "All",
                "bsc_std": "All_std",
            }
        )
        scs_data_df = self._sort_model_names(data_df.iloc[:, 6:12]).rename(
            columns={
                "p_scs": "Sarcastic",
                "p_scs_std": "Sarcastic_std",
                "n_scs": "Non-Sarcastic",
                "n_scs_std": "Non-Sarcastic_std",
                "scs": "All",
                "scs_std": "All_std",
            }
        )
        lcs_data_df = self._sort_model_names(data_df.iloc[:, 12:18]).rename(
            columns={
                "p_lcs": "Sarcastic",
                "p_lcs_std": "Sarcastic_std",
                "n_lcs": "Non-Sarcastic",
                "n_lcs_std": "Non-Sarcastic_std",
                "lcs": "All",
                "lcs_std": "All_std",
            }
        )
        tsc_data_df = self._sort_model_names(data_df.iloc[:, 18:]).rename(
            columns={
                "sarc_tsc": "Sarcastic",
                "sarc_tsc_std": "Sarcastic_std",
                "non_sarc_tsc": "Non-Sarcastic",
                "non_sarc_tsc_std": "Non-Sarcastic_std",
                "neutral_tsc": "Neutral",
                "neutral_tsc_std": "Neutral_std",
                "tsc": "All",
                "tsc_std": "All_std",
            }
        )

        sns.set_theme(context="talk", style="white", font="Arial", font_scale=2)
        fig, axes = plt.subplots(1, 4, figsize=(34, 12), sharey=True)
        for i, (task, data_df) in enumerate(
            zip(
                ["BSC", "TSC", "SCS", "LCS"],
                [bsc_data_df, tsc_data_df, scs_data_df, lcs_data_df],
            )
        ):
            ax = axes[i]

            classes = ["Sarcastic", "Non-Sarcastic", "All"]
            if task == "TSC":
                classes = ["Sarcastic", "Neutral", "Non-Sarcastic", "All"]
            df = data_df.reset_index().rename(columns={"index": "Model"})
            df_long = pd.DataFrame()

            for cls in classes:
                if cls in df.columns and f"{cls}_std" in df.columns:
                    temp_df = df[["Model", cls, f"{cls}_std"]].copy()
                    temp_df.rename(
                        columns={cls: "Score", f"{cls}_std": "Std"}, inplace=True
                    )
                    temp_df["Prediction"] = cls
                    df_long = pd.concat([df_long, temp_df], ignore_index=True)

            color_palette = sns.color_palette(
                "Set2", len(classes)
            )  # Using 'Set2' for distinct colors

            # Drop rows with NaN in the "Score" column
            df_long = df_long.dropna(subset=["Score"])
            # Create a horizontal barplot (Model on the y-axis, Score on the x-axis)
            bar_plot = sns.barplot(
                data=df_long,
                y="Model",
                x="Score",
                hue="Prediction",
                palette=color_palette,
                orient="h",  # Explicitly define horizontal orientation
                edgecolor="black",  # Add a border to each bar
                linewidth=1.5,  # Set the border width
                ax=ax,
            )
            for j, bar in enumerate(bar_plot.patches):
                if j >= len(df_long):
                    break
                # Calculate the center of the bar on the y-axis
                y_center = bar.get_y() + bar.get_height() / 2

                # Current bar's Score is its width
                # We use 'Std' from the df_long to determine the error range
                std = df_long["Std"].iloc[j]
                score = df_long["Score"].iloc[j]

                # Plot the error bar horizontally
                bar_plot.errorbar(
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

            ax.set_title(task)
            ax.set_xlabel("NLL")
            ax.legend(
                prop={"size": 20},
                title_fontsize=22,
                framealpha=0.75,
                loc="lower left",
                title="Prediction",
            )

        # fig.legend(
        #     loc="center right",
        #     bbox_to_anchor=(1.06, 0.5),
        #     labels=["Sarcastic", "Non-Sarcastic", "All", "Neutral"],
        #     title="Prediction",
        # )

        fig.savefig(
            str(self.output_path / "model_nll_analysis.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
