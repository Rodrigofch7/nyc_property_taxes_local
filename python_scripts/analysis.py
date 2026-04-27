import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

DATA_PATH  = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/merged_fairness_2020_2024.parquet"
ASSESS_BASE = "/mnt/c/Users/rodri/Documents/NYC Datasets/assessment_interim"
OUTPUT_DIR = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

TAX_CLASS_LABELS = {1.0: "Class 1\n(1-3 family)", 2.0: "Class 2\n(apartments/coops)", 4.0: "Class 4\n(commercial)"}
BLDG_CLASS_MAP = {
    "A": "1-3 Family", "B": "2-Family", "C": "Walk-up Apt",
    "D": "Elevator Apt", "R": "Condo", "S": "Mixed Res/Comm",
    "K": "Retail", "O": "Office", "H": "Hotel",
}

df["TAX_CLASS"]  = df["TAX CLASS AT TIME OF SALE"].map(TAX_CLASS_LABELS)
df["BLDG_TYPE"]  = df["BLDG_CLASS"].str[0].map(BLDG_CLASS_MAP).fillna("Other")
BOROUGH_MAP = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}
df["BOROUGH_NAME"] = df["BOROUGH"].map(BOROUGH_MAP)

def cod(s):
    med = s.median()
    return (s - med).abs().mean() / med * 100 if med != 0 else np.nan

# ── 1. Sales ratio distributions by tax class ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle("Sales Ratio Distribution by Tax Class", fontsize=14, fontweight="bold", y=1.02)

colors = {1.0: "#378ADD", 2.0: "#D85A30", 4.0: "#1D9E75"}
for ax, tc in zip(axes, [1.0, 2.0, 4.0]):
    subset = df[df["TAX CLASS AT TIME OF SALE"] == tc]["sales_ratio"].clip(0, 4)
    ax.hist(subset, bins=80, color=colors[tc], alpha=0.85, edgecolor="none")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="Fair assessment (1.0)")
    ax.axvline(subset.median(), color="red", linestyle="-", linewidth=1.2, label=f"Median: {subset.median():.2f}")
    ax.set_title(TAX_CLASS_LABELS[tc], fontsize=11)
    ax.set_xlabel("Sales ratio (city FMV / sale price)")
    ax.set_ylabel("Number of sales")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_sales_ratio_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 1_sales_ratio_distribution.png")

# ── 2. Class 2 breakdown by building type ─────────────────────────────────────
class2 = df[df["TAX CLASS AT TIME OF SALE"] == 2.0].copy()

class2_summary = (
    class2.groupby("BLDG_TYPE")["sales_ratio"]
    .agg(count="count", median_ratio="median", cod=cod)
    .query("count >= 50")
    .sort_values("median_ratio")
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(class2_summary["BLDG_TYPE"], class2_summary["median_ratio"],
               color="#D85A30", alpha=0.85)
ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="Fair (1.0)")
for bar, (_, row) in zip(bars, class2_summary.iterrows()):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"n={row['count']:,}  COD={row['cod']:.0f}", va="center", fontsize=8)
ax.set_xlabel("Median sales ratio (city FMV / sale price)")
ax.set_title("Class 2: Median sales ratio by building type", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_class2_by_building_type.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 2_class2_by_building_type.png")

# ── 3. Sales ratio by borough ─────────────────────────────────────────────────
borough_class = (
    df.groupby(["BOROUGH_NAME", "TAX CLASS AT TIME OF SALE"])["sales_ratio"]
    .median()
    .reset_index()
    .pivot(index="BOROUGH_NAME", columns="TAX CLASS AT TIME OF SALE", values="sales_ratio")
    .rename(columns={1.0: "Class 1", 2.0: "Class 2", 4.0: "Class 4"})
)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(borough_class))
width = 0.25
for i, (col, color) in enumerate(zip(["Class 1", "Class 2", "Class 4"],
                                      ["#378ADD", "#D85A30", "#1D9E75"])):
    if col in borough_class.columns:
        ax.bar(x + i * width, borough_class[col], width, label=col, color=color, alpha=0.85)

ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Fair (1.0)")
ax.set_xticks(x + width)
ax.set_xticklabels(borough_class.index)
ax.set_ylabel("Median sales ratio")
ax.set_title("Median sales ratio by borough and tax class", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_sales_ratio_by_borough.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 3_sales_ratio_by_borough.png")

# ── 4. Cap effect — PYACTTOT vs FINACTTOT (Class 2 only) ─────────────────────
py_av = pd.read_parquet(
    f"{ASSESS_BASE}/assessment_FY2024.parquet",
    columns=["BBL", "PYACTTOT"]
).drop_duplicates(subset="BBL")

py_av["PYACTTOT_clean"] = (
    py_av["PYACTTOT"].astype(str)
    .str.replace(r"[+\s]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce")
)

class2 = class2.merge(py_av[["BBL", "PYACTTOT_clean"]], on="BBL", how="left")

class2["AV_change_pct"] = (
    (class2["CURRENT_FINACTTOT"] - class2["PYACTTOT_clean"])
    / class2["PYACTTOT_clean"] * 100
)

cap_data = class2[
    (class2["AV_change_pct"].between(-5, 5)) &
    (class2["sales_ratio"] < 0.5)
].copy()

cap_pct = len(cap_data) / len(class2) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Class 2: Assessment cap effect", fontsize=13, fontweight="bold")

axes[0].hist(class2["AV_change_pct"].clip(-30, 30), bins=60,
             color="#D85A30", alpha=0.85, edgecolor="none")
axes[0].axvline(8, color="black", linestyle="--", linewidth=1.2, label="8% annual cap")
axes[0].axvline(0, color="gray", linestyle=":", linewidth=1.0)
axes[0].set_xlabel("YoY assessed value change (%)")
axes[0].set_ylabel("Number of properties")
axes[0].set_title("AV year-over-year change")
axes[0].legend(fontsize=9)

sample = class2[class2["AV_change_pct"].between(-30, 30)].sample(min(5000, len(class2)), random_state=42)
axes[1].scatter(sample["AV_change_pct"], sample["sales_ratio"].clip(0, 3),
                alpha=0.15, s=8, color="#D85A30")
axes[1].axvline(8, color="black", linestyle="--", linewidth=1.0, label="8% cap")
axes[1].axhline(1.0, color="blue", linestyle="--", linewidth=1.0, label="Fair ratio")
axes[1].set_xlabel("YoY AV change (%)")
axes[1].set_ylabel("Sales ratio")
axes[1].set_title(f"Cap effect: {cap_pct:.1f}% of Class 2 sales\nhave frozen AV + ratio < 0.5")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_cap_effect_class2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 4_cap_effect_class2.png")

# ── Print summary tables ──────────────────────────────────────────────────────
print("\n── Class 2 breakdown by building type ──")
print(class2_summary.to_string(index=False))

print("\n── Median sales ratio by borough & class ──")
print(borough_class.round(3).to_string())

print(f"\n── Cap effect: {cap_pct:.1f}% of Class 2 properties appear AV-capped ──")