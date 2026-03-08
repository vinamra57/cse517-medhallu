import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "medqa_hallucinated.csv"          # change if your file lives elsewhere
OUTPUT_PATH = "hallucination_difficulty_chart.png"

DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
CATEGORY_ORDER    = [
    'Misinterpretation of Question',
    'Incomplete Information',
    "Mechanism and Pathway Misattribution",
    "Methodological and Evidence Fabrication"
]

# Colour palette – one shade per difficulty level
COLORS = {
    "Easy":   "#4CAF8A",   # teal-green
    "Medium": "#F4A836",   # amber
    "Hard":   "#E05C5C",   # coral-red
}

BAR_WIDTH  = 0.22
GROUP_GAP  = 0.72          # centre-to-centre distance between category groups


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. "
            "Run main.py first, or point CSV_PATH at your results file."
        )
    df = pd.read_csv(csv_path)

    required = {"Category of Hallucination", "Difficulty Level"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df["Category of Hallucination"] = df["Category of Hallucination"].str.strip()
    df["Difficulty Level"]          = df["Difficulty Level"].str.strip().str.capitalize()

    # Anything not in the known difficulty list → 'Unknown difficulty'
    df.loc[~df["Difficulty Level"].isin(DIFFICULTY_LEVELS), "Difficulty Level"] = "Easy"

    # Anything not in the known category list → 'Unknown'
    df.loc[~df["Category of Hallucination"].isin(CATEGORY_ORDER), "Category of Hallucination"] = "Unknown"

    return df


def compute_percentages(df: pd.DataFrame) -> dict:
    """
    Returns {category: {difficulty: pct_of_that_category}} 
    pct sums to 100 across the three difficulty bars for each category.
    """
    data = {}
    for cat in CATEGORY_ORDER:
        subset = df[df["Category of Hallucination"] == cat]
        total  = len(subset)
        data[cat] = {}
        for diff in DIFFICULTY_LEVELS:
            count = len(subset[subset["Difficulty Level"] == diff])
            data[cat][diff] = (count / total * 100) if total > 0 else 0.0
    return data


def make_chart(data: dict, n_samples: int, output_path: str, df):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("#F7F8FA")
    ax.set_facecolor("#F7F8FA")

    n_cats    = len(CATEGORY_ORDER)
    x_centres = np.arange(n_cats) * GROUP_GAP
    offsets   = np.array([-BAR_WIDTH, 0, BAR_WIDTH])

    for d_idx, diff in enumerate(DIFFICULTY_LEVELS):
        xpos   = x_centres + offsets[d_idx]
        values = [data[cat][diff] for cat in CATEGORY_ORDER]

        bars = ax.bar(
            xpos, values,
            width=BAR_WIDTH - 0.02,
            color=COLORS[diff],
            label=diff,
            zorder=3,
            linewidth=0,
        )

        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.0f}%",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#444",
                    fontweight="bold",
                )

    for cat_idx, cat in enumerate(CATEGORY_ORDER):
        total = df[df["Category of Hallucination"] == cat].shape[0]
        ax.text(
            x_centres[cat_idx],
            105,
            f"Total: {total}",
            ha="center", va="bottom",
            fontsize=8.5, color="#555",
            fontstyle="italic",
        )
    # Axes formatting
    ax.set_xticks(x_centres)
    ax.set_xticklabels(CATEGORY_ORDER, fontsize=10.5, color="#333", rotation=20, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage of Category Samples (%)", fontsize=11, color="#444", labelpad=10)
    ax.set_xlabel("Hallucination Category", fontsize=11, color="#444", labelpad=10)
    ax.set_title(
        "Difficulty Distribution Across Hallucination Categories",
        fontsize=14, fontweight="bold", color="#222", pad=18,
    )

    # Subtle horizontal grid
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#ddd", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(axis="both", colors="#555")

    # Legend
    legend = ax.legend(
        title="Difficulty", title_fontsize=10,
        fontsize=9.5, framealpha=0.9,
        loc="upper right", edgecolor="#ccc",
    )
    legend.get_frame().set_linewidth(0.8)

    # Footer note
    fig.text(
        0.99, 0.01,
        f"n = {n_samples} total samples",
        ha="right", va="bottom",
        fontsize=8, color="#999",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved → {output_path}")
    plt.show()

def get_accuracy(df, df_actual):
    accurate = 0
    for i in range(len(df)):
        diff_1 = df["Difficulty Level"][i]
        diff_2 = df_actual["Difficulty Level"][i]

        if diff_1.lower() == diff_2.lower():
            accurate += 1
    return accurate/len(df)


def main():
    print(f"Loading data from '{CSV_PATH}' …")
    df = load_data(CSV_PATH)
    df_actual = load_data("medhallu_dataset_artificial.csv")[:500]

    accuracy = get_accuracy(df, df_actual)
    print(accuracy)

    print(f"  {len(df)} rows loaded")
    print(f"  Categories found: {df['Category of Hallucination'].value_counts().to_dict()}")
    print(f"  Difficulties found: {df['Difficulty Level'].value_counts().to_dict()}")

    data = compute_percentages(df)
    make_chart(data, n_samples=len(df), output_path=OUTPUT_PATH, df = df)


if __name__ == "__main__":
    main()