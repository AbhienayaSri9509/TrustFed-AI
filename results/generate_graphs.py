import os
import json
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METRICS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)

FIGURES_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures"
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ============================================================
# LOAD ABLATION RESULTS
# ============================================================

csv_path = os.path.join(
    METRICS_DIR,
    "ablation_results.csv"
)

df = pd.read_csv(
    csv_path
)

print("\nLoaded ablation results:")
print(df.to_string(index=False))


# ============================================================
# SHORT NAMES
# ============================================================

short_names = [
    "FedAvg",
    "Trust",
    "Trust+Resource",
    "Trust+Resource+Network",
    "TPRS",
    "Full TrustFed"
]

df["ShortName"] = short_names


# ============================================================
# FIGURE 1 — ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    df["ShortName"],
    df["Accuracy"],
    marker="o"
)

plt.xlabel(
    "Ablation Configuration"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Ablation Study: Accuracy Comparison"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "ablation_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 2 — PRECISION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    df["ShortName"],
    df["Precision"],
    marker="o"
)

plt.xlabel(
    "Ablation Configuration"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Ablation Study: Precision Comparison"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "ablation_precision.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 3 — RECALL
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    df["ShortName"],
    df["Recall"],
    marker="o"
)

plt.xlabel(
    "Ablation Configuration"
)

plt.ylabel(
    "Recall"
)

plt.title(
    "Ablation Study: Recall Comparison"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "ablation_recall.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 4 — F1 SCORE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    df["ShortName"],
    df["F1"],
    marker="o"
)

plt.xlabel(
    "Ablation Configuration"
)

plt.ylabel(
    "F1 Score"
)

plt.title(
    "Ablation Study: F1 Score Comparison"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "ablation_f1.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 5 — ALL METRICS
# ============================================================

plt.figure(
    figsize=(11, 7)
)

plt.plot(
    df["ShortName"],
    df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    df["ShortName"],
    df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    df["ShortName"],
    df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    df["ShortName"],
    df["F1"],
    marker="o",
    label="F1"
)

plt.xlabel(
    "Ablation Configuration"
)

plt.ylabel(
    "Score"
)

plt.title(
    "TrustFed-6G Ablation Study"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "ablation_all_metrics.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SAVE SUMMARY TABLE
# ============================================================

summary = df[
    [
        "ShortName",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
].copy()

summary.columns = [
    "Configuration",
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

summary.to_csv(
    os.path.join(
        FIGURES_DIR,
        "final_ablation_table.csv"
    ),
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print(
    "E7 GRAPH GENERATION COMPLETE"
)

print("=" * 70)

print(
    "\nFigures saved in:"
)

print(
    FIGURES_DIR
)

print("\nGenerated files:")

for filename in os.listdir(
    FIGURES_DIR
):

    print(
        " -",
        filename
    )