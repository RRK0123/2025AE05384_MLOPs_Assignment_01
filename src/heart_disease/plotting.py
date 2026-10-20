"""Generate EDA figures with a headless plotting backend."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .schema import CATEGORIES, DISPLAY, LABELS, NUMERIC, TARGET

PALETTE = {"Disease absent": "#247A89", "Disease present": "#C55B40"}


def wilson_interval(positive: int, total: int) -> tuple[float, float]:
    """95% Wilson interval for descriptive within-group proportions."""
    if total == 0:
        return float("nan"), float("nan")
    z = 1.96
    proportion = positive / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0, center - radius), min(1, center + radius)


def save_figure(fig, directory, name, title, subtitle):
    fig.suptitle(title, x=0.04, ha="left", fontsize=19, fontweight="bold", y=1.03)
    fig.text(0.04, 0.005, subtitle, fontsize=10, color="#52616B")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(directory / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_figures(raw, cleaned, train, figures, tables):
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    scope = f"Training subset only (n={len(train)}); observed values, before imputation."

    missing = raw.isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = missing["missing_count"] / len(raw) * 100
    missing.to_csv(tables / "missing_values.csv", index_label="column")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(missing.index, missing["missing_count"], color="#247A89")
    ax.set(
        ylabel="Missing observations",
        xlabel="Original source column",
        ylim=(0, max(1, missing["missing_count"].max()) + 1),
    )
    for i, count in enumerate(missing["missing_count"]):
        ax.text(i, count + 0.08, str(count), ha="center", fontsize=10)
    save_figure(
        fig,
        figures,
        "01_missing_values.png",
        "Where is the source data incomplete?",
        f"Full-source quality audit (n={len(raw)}); '?' interpreted as missing.",
    )

    counts = cleaned[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [LABELS[x] for x in counts.index], counts, color=list(PALETTE.values()), width=0.55
    )
    ax.set(ylabel="Number of observations", ylim=(0, counts.max() * 1.2))
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 3,
            f"{count}\n({count / len(cleaned):.1%})",
            ha="center",
        )
    save_figure(
        fig,
        figures,
        "02_class_balance.png",
        "How balanced is the binary target?",
        f"Full-source target audit (n={len(cleaned)}); original diagnosis 1-4 mapped to disease present.",
    )

    plot_data = train.assign(outcome=train[TARGET].map(LABELS))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, column in zip(axes.flat, NUMERIC):
        sns.histplot(
            data=plot_data,
            x=column,
            hue="outcome",
            hue_order=list(PALETTE),
            palette=PALETTE,
            bins=15,
            element="step",
            alpha=0.32,
            common_norm=False,
            ax=ax,
        )
        ax.set(xlabel=DISPLAY[column], ylabel="Count")
        if column != NUMERIC[0] and ax.get_legend() is not None:
            ax.get_legend().remove()
    axes.flat[-1].axis("off")
    save_figure(
        fig,
        figures,
        "03_numeric_histograms.png",
        "How do measurements vary across outcomes?",
        scope,
    )

    correlation = train[NUMERIC].corr(method="pearson")
    correlation.to_csv(tables / "numeric_correlations.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"label": "Pearson correlation"},
    )
    ax.tick_params(axis="x", rotation=30)
    save_figure(
        fig,
        figures,
        "04_numeric_correlation.png",
        "Which continuous measurements move together?",
        scope + " Category codes and target excluded from Pearson correlations.",
    )

    fig, axes = plt.subplots(4, 2, figsize=(14, 17))
    category_rows = []
    for ax, (column, mapping) in zip(axes.flat, CATEGORIES.items()):
        labels, rates, lower, upper, totals = [], [], [], [], []
        for level, label in mapping.items():
            observed = train.loc[train[column] == level, TARGET]
            total, positive = len(observed), int(observed.sum())
            if total == 0:
                continue
            low, high = wilson_interval(positive, total)
            rate = positive / total
            labels.append(label)
            rates.append(rate * 100)
            lower.append((rate - low) * 100)
            upper.append((high - rate) * 100)
            totals.append(total)
            category_rows.append(
                {
                    "feature": column,
                    "code": level,
                    "label": label,
                    "n": total,
                    "positive": positive,
                    "disease_percent": rate * 100,
                    "wilson_low_percent": low * 100,
                    "wilson_high_percent": high * 100,
                }
            )
        positions = np.arange(len(labels))
        ax.barh(positions, rates, color="#247A89", height=0.6)
        ax.errorbar(rates, positions, xerr=[lower, upper], fmt="none", ecolor="#23343E", capsize=4)
        ax.set(
            yticks=positions,
            yticklabels=labels,
            xlim=(0, 115),
            xlabel="Disease present (%)",
            title=column,
        )
        ax.set_xticks([0, 25, 50, 75, 100])
        for y, rate, high, total in zip(positions, rates, upper, totals):
            ax.text(min(rate + high + 2, 102), y, f"n={total}", va="center", fontsize=9)
    pd.DataFrame(category_rows).to_csv(tables / "categorical_disease_rates.csv", index=False)
    save_figure(
        fig,
        figures,
        "05_categorical_disease_rates.png",
        "How does observed disease frequency vary by category?",
        scope
        + " Bars: observed proportions; whiskers: descriptive 95% Wilson intervals. Associations are not causal.",
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, column in zip(axes.flat, NUMERIC):
        sns.boxplot(
            data=plot_data,
            x="outcome",
            y=column,
            hue="outcome",
            order=list(PALETTE),
            hue_order=list(PALETTE),
            palette=PALETTE,
            legend=False,
            ax=ax,
        )
        ax.set(xlabel="", ylabel=DISPLAY[column])
        ax.tick_params(axis="x", labelsize=9)
    axes.flat[-1].axis("off")
    save_figure(
        fig,
        figures,
        "06_numeric_boxplots.png",
        "Where do the outcome groups overlap?",
        scope + " Points outside boxplot whiskers are retained.",
    )

    outliers = []
    for column in NUMERIC:
        values = train[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = (values < low) | (values > high)
        outliers.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "lower_fence": low,
                "upper_fence": high,
                "flagged_count": int(flagged.sum()),
                "flagged_percent": float(flagged.mean() * 100),
            }
        )
    outlier_table = pd.DataFrame(outliers)
    outlier_table.to_csv(tables / "iqr_outlier_audit.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(outlier_table["feature"], outlier_table["flagged_count"], color="#C55B40")
    ax.set(ylabel="Observations outside 1.5 x IQR fences", xlabel="Continuous measurement")
    for i, count in enumerate(outlier_table["flagged_count"]):
        ax.text(i, count + 0.15, str(count), ha="center")
    ax.set_ylim(0, max(1, outlier_table["flagged_count"].max()) + 2)
    save_figure(
        fig,
        figures,
        "07_outlier_audit.png",
        "Which measurements have unusually extreme values?",
        scope + " Statistical flags only; no rows removed or values clipped.",
    )
