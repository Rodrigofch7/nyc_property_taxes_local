"""
app.py
======
Streamlit app for NYC Property Tax Assessment Classification
Three tabs:
  1. Methodology — model explanation, feature importance, confusion matrix
  2. Borough Analysis — charts of assessment patterns
  3. BBL Lookup — enter a BBL and get a classification with peer comparison

Deploy: streamlit run app.py
"""

import os
import sys
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))  # app_deploy/
PROJECT_DIR = os.path.dirname(BASE_DIR)                      # project root
sys.path.insert(0, os.path.join(PROJECT_DIR, "python_scripts"))

DATA_PATH  = os.path.join(PROJECT_DIR, "data",    "processed_labeled_data.parquet")
MODEL_DIR  = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# ── Page config ───────────────────────────────────────────────────────────────
icon_path = os.path.join(BASE_DIR, "nyc.png")
st.set_page_config(
    page_title="NYC Property Tax Assessment",
    page_icon=icon_path if os.path.exists(icon_path) else "🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "overvalued":    "#d73027",
    "undervalued":   "#4575b4",
    "fairly_valued": "#1a9850",
}
CLASS_LABELS = {
    "overvalued":    "🔴 Overvalued",
    "undervalued":   "🔵 Undervalued",
    "fairly_valued": "🟢 Fairly Valued",
}
BORO_MAP = {"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens", "5": "Staten Island"}

# Columns to keep in memory — only what the app actually uses
KEEP_COLS = [
    "BBL", "BORO", "BLDG_CLASS", "ZIP_CODE", "GROSS_SQFT", "LAND_AREA",
    "YRBUILT", "UNITS", "NUM_BLDGS", "BLD_STORY", "ZONING",
    "FINACTTOT", "FINACTLAND", "FINMKTTOT",
    "target_2026",
    "FINACTTOT_FY2020", "FINACTTOT_FY2021", "FINACTTOT_FY2022",
    "FINACTTOT_FY2023", "FINACTTOT_FY2024", "FINACTTOT_FY2025",
    "overvalued_2020",    "overvalued_2021",    "overvalued_2022",
    "overvalued_2023",    "overvalued_2024",    "overvalued_2025",
    "undervalued_2020",   "undervalued_2021",   "undervalued_2022",
    "undervalued_2023",   "undervalued_2024",   "undervalued_2025",
    "fairly_valued_2020", "fairly_valued_2021", "fairly_valued_2022",
    "fairly_valued_2023", "fairly_valued_2024", "fairly_valued_2025",
]


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model    = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    le_dict  = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    return model, features, le_dict


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    cols = [c for c in KEEP_COLS]  # parquet columnar read — only what we need
    try:
        df = pd.read_parquet(DATA_PATH, columns=cols)
    except Exception:
        # fallback: read all and drop extras
        df = pd.read_parquet(DATA_PATH)
        df = df[[c for c in cols if c in df.columns]]
    df["BBL"]       = df["BBL"].astype(str).str.strip()
    df["BORO_NAME"] = df["BORO"].astype(str).map(BORO_MAP).fillna("Unknown")
    # numeric coercions up front
    for col in ["FINACTTOT", "GROSS_SQFT", "FINMKTTOT", "FINACTLAND"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_feature_importance():
    path = os.path.join(OUTPUT_DIR, "lgbm_feature_importance.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


# ── Prediction: uses pre-computed target from dataset (no re-engineering) ─────
def lookup_bbl(bbl: str, df):
    row = df[df["BBL"] == bbl.strip()]
    return None if row.empty else row.iloc[0]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists(icon_path):
        st.image(icon_path, width=80)
    st.title("NYC Property Tax")
    st.markdown("**Assessment Classification Model**")
    st.markdown("---")
    st.markdown("""
    **LightGBM** trained on NYC DOF assessment data (FY2020–FY2026)
    to classify ~1.1M properties as:

    - 🔴 **Overvalued** — assessed >15% above peer median
    - 🟢 **Fairly Valued** — within ±15% of peer median
    - 🔵 **Undervalued** — assessed >15% below peer median

    Peer groups: borough + building class (up to 6-level fallback hierarchy).
    """)
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.metric("Test F1 Macro",       "87.9%")
    st.metric("Test Accuracy",       "88.2%")
    st.metric("Training Properties", "877k")
    st.markdown("---")
    st.caption("CAPP 30254 · Spring 2026 · UChicago  \nAhmed Lodhi · Faizan Imran · Rodrigo Chaves")


# ── Load ──────────────────────────────────────────────────────────────────────
model, features, le_dict = load_model()
df       = load_data()
feat_imp = load_feature_importance()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Methodology", "🗺️ Borough Analysis", "🔍 BBL Lookup"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — METHODOLOGY
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Methodology")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Research Question")
        st.markdown("""
        Can we classify NYC properties as **undervalued**, **fairly valued**, or **overvalued**
        based on assessed value per square foot relative to peer properties,
        using structural, geographic, and historical assessment features?
        """)

        st.subheader("Data")
        st.markdown("""
        - **Source:** NYC Department of Finance Property Assessment Rolls
        - **Years:** FY2020–FY2026 (FY2015–FY2022 also collected)
        - **Properties:** ~1.1 million tax lots (all 5 boroughs, all tax classes)
        - **Sales data:** NYC DOF Annualized Sales 2015–2024 (~571k arm's-length transactions)
        - **Target:** FY2026 peer-group classification (±15% threshold)
        """)

        st.subheader("Labeling Strategy")
        st.markdown("""
        Each property's **assessed value per square foot** is compared to its
        **peer group median** using a 5-level fallback hierarchy
        (coarsens automatically if group size < 10):

        1. Borough + Building Class + Tax Class + Decade Built + Size Bin + Market Value Bin
        2. Drop size bin
        3. Drop decade bin
        4. Drop market value bin
        5. Borough + Building Class *(coarsest fallback)*

        | Label | Condition |
        |---|---|
        | Overvalued | > 115% of peer median |
        | Fairly Valued | 85–115% of peer median |
        | Undervalued | < 85% of peer median |

        Historical labels (FY2020–FY2025) become **features** for predicting FY2026.
        """)

        st.subheader("Model")
        st.markdown("""
        **LightGBM** (gradient boosting, leaf-wise tree growth)
        - Tuned via 20-iteration checkpointed random search × 5-fold CV on 300k subsample
        - `class_weight="balanced"` to handle class imbalance
        - Best params: `n_estimators=800, num_leaves=511, learning_rate=0.05`
        - **Primary metric:** Macro F1 (equal weight per class)
        """)

    with col2:
        st.subheader("Model Performance")
        perf_df = pd.DataFrame({
            "Model":          ["SGD L2", "SGD ElasticNet", "SGD L1", "LightGBM"],
            "Type":           ["Linear", "Linear", "Linear", "Non-linear"],
            "Test F1 Macro":  [0.8149, 0.8147, 0.7520, 0.8790],
            "Test Accuracy":  [0.8188, 0.8182, 0.7584, 0.8822],
            "CV F1 Macro":    [0.8135, 0.8136, 0.7581, None],
        })
        st.dataframe(
            perf_df.style.highlight_max(
                subset=["Test F1 Macro", "Test Accuracy"],
                color="#c6efce"
            ).format({
                "Test F1 Macro": "{:.4f}",
                "Test Accuracy": "{:.4f}",
                "CV F1 Macro":   lambda x: f"{x:.4f}" if pd.notna(x) else "—",
            }),
            use_container_width=True,
        )
        st.caption("Baseline (majority class): 0.527  |  Linear models use PCA (60 components) + StandardScaler")

        st.subheader("Top Features (LightGBM)")
        if feat_imp is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            top20 = feat_imp.head(20)
            ax.barh(top20["Feature"][::-1], top20["Importance"][::-1], color="#4575b4")
            ax.set_xlabel("Importance")
            ax.set_title("Top 20 Feature Importances")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Feature importance CSV not found in outputs/.")

        cm_path = os.path.join(OUTPUT_DIR, "lgbm_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.subheader("Confusion Matrix")
            st.image(cm_path, use_column_width=True)

    st.subheader("Feature Groups (138 total)")
    feat_groups = {
        "📍 Location":            ["BORO_CODE", "ZIP_CODE_CODE", "ZONING_CODE", "BLDG_CLASS_CODE",
                                    "ZIP_MEAN_ASSESS", "ZIP_ASSESS_STD", "ASSESS_VS_ZIP_MEDIAN"],
        "🏗️ Physical":            ["LOG_GROSS_SQFT", "LOG_LAND_AREA", "BUILDING_AGE", "BUILDING_ERA",
                                    "SQFT_PER_UNIT", "COVERAGE_RATIO"],
        "💰 Assessment (FY2025)": ["LOG_ASSESS_PER_SQFT", "MKT_TO_ASSESS", "LOG_MKT_TO_ASSESS",
                                    "ASSESS_VS_BLDG_CLASS_MEDIAN", "ASSESS_VS_ZIP_MEDIAN"],
        "📈 Trend & Momentum":    ["ASSESS_TREND", "ASSESS_VOLATILITY", "ASSESS_AT_CAP",
                                    "MKT_TREND", "MKT_VS_ASSESS_TREND_SPREAD"],
        "🔄 YoY & Acceleration":  ["ASSESS_YOY_FY2021–2025", "LAND_YOY_FY2021–2025",
                                    "ASSESS_ACCEL_FY2022–2025"],
        "📊 Historical Status":   ["overvalued_2020–2025", "undervalued_2020–2025",
                                    "CONSISTENT_OVERVALUED", "DOMINANT_CLASS"],
        "🔮 OLS Projections":     ["PROJ_FINACTTOT_FY2026", "PROJ_RATIO_FINACTTOT_FY2026",
                                    "PROJ_RESID_FINACTTOT_FY2026"],
        "🏠 Sales (BBL join)":    ["LAST_SALE_PRICE", "SALE_TO_ASSESS_RATIO",
                                    "SALE_PRICE_PER_SQFT", "YEARS_SINCE_SALE"],
        "✖️ Interactions":        ["AGE_X_ASSESS_PER_SQFT"],
    }
    cols = st.columns(3)
    for i, (group, feats) in enumerate(feat_groups.items()):
        with cols[i % 3]:
            st.markdown(f"**{group}**")
            for f in feats:
                st.markdown(f"  - `{f}`")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — BOROUGH ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Borough Analysis")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_boros = st.multiselect(
            "Filter by Borough",
            options=list(BORO_MAP.values()),
            default=list(BORO_MAP.values()),
        )
    with col_f2:
        selected_classes = st.multiselect(
            "Filter by Classification",
            options=["overvalued", "fairly_valued", "undervalued"],
            default=["overvalued", "fairly_valued", "undervalued"],
            format_func=lambda x: CLASS_LABELS[x],
        )

    df_filtered = df[
        df["BORO_NAME"].isin(selected_boros) &
        df["target_2026"].isin(selected_classes)
    ]
    st.caption(f"Showing {len(df_filtered):,} properties")

    m1, m2, m3, m4 = st.columns(4)
    vc = df_filtered["target_2026"].value_counts()
    total = max(len(df_filtered), 1)
    m1.metric("🔴 Overvalued",    f"{vc.get('overvalued',    0):,}", f"{vc.get('overvalued',    0)/total*100:.1f}%")
    m2.metric("🟢 Fairly Valued", f"{vc.get('fairly_valued', 0):,}", f"{vc.get('fairly_valued', 0)/total*100:.1f}%")
    m3.metric("🔵 Undervalued",   f"{vc.get('undervalued',   0):,}", f"{vc.get('undervalued',   0)/total*100:.1f}%")
    m4.metric("📦 Total",         f"{len(df_filtered):,}")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("Classification by Borough")
        boro_pivot = (
            df_filtered.groupby(["BORO_NAME", "target_2026"]).size()
            .reset_index(name="count")
            .pivot(index="BORO_NAME", columns="target_2026", values="count")
            .fillna(0)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        boro_pivot.plot(kind="bar", ax=ax, stacked=True,
                        color=[CLASS_COLORS.get(c, "#999") for c in boro_pivot.columns])
        ax.set_xlabel("")
        ax.set_ylabel("Properties")
        ax.set_title("Classification by Borough")
        ax.legend(title="Class", bbox_to_anchor=(1.05, 1), fontsize=8)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_c2:
        st.subheader("Top 15 Building Classes")
        top_bldg = df_filtered["BLDG_CLASS"].value_counts().head(15).index.tolist()
        bldg_pivot = (
            df_filtered[df_filtered["BLDG_CLASS"].isin(top_bldg)]
            .groupby(["BLDG_CLASS", "target_2026"]).size()
            .reset_index(name="count")
            .pivot(index="BLDG_CLASS", columns="target_2026", values="count")
            .fillna(0)
        )
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        bldg_pivot.plot(kind="bar", ax=ax2, stacked=True,
                        color=[CLASS_COLORS.get(c, "#999") for c in bldg_pivot.columns])
        ax2.set_xlabel("")
        ax2.set_ylabel("Properties")
        ax2.set_title("Top 15 Building Classes")
        ax2.legend(title="Class", bbox_to_anchor=(1.05, 1), fontsize=8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.subheader("Assessed Value per Sqft by Borough")
    df_plot = df_filtered.copy()
    df_plot["assess_per_sqft"] = (
        df_plot["FINACTTOT"] / df_plot["GROSS_SQFT"].clip(lower=1)
    )
    df_plot = df_plot[df_plot["assess_per_sqft"].between(0, 500)]
    if not df_plot.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        for boro in selected_boros:
            data = df_plot[df_plot["BORO_NAME"] == boro]["assess_per_sqft"].dropna()
            if not data.empty:
                ax3.hist(data, bins=60, alpha=0.5, label=boro, density=True)
        ax3.set_xlabel("Assessed Value per Sqft ($)")
        ax3.set_ylabel("Density")
        ax3.set_title("Distribution of Assessed Value per Sqft")
        ax3.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.subheader("Summary by Borough")
    summary = (
        df_filtered.groupby("BORO_NAME")
        .agg(
            Total=("BBL", "count"),
            Overvalued=("target_2026", lambda x: (x == "overvalued").sum()),
            Fairly_Valued=("target_2026", lambda x: (x == "fairly_valued").sum()),
            Undervalued=("target_2026", lambda x: (x == "undervalued").sum()),
        )
        .reset_index()
    )
    summary["% Over"]  = (summary["Overvalued"]    / summary["Total"] * 100).round(1)
    summary["% Fair"]  = (summary["Fairly_Valued"] / summary["Total"] * 100).round(1)
    summary["% Under"] = (summary["Undervalued"]   / summary["Total"] * 100).round(1)
    st.dataframe(summary, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — BBL LOOKUP
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🔍 Property Lookup")
    st.markdown("""
    Enter a **BBL (Borough-Block-Lot)** to look up a property's FY2026 classification
    and see how it compares to its peer group.

    Format: `BBBBBBBLLLL` (10 digits) — e.g. `1000750043` for Manhattan.
    """)

    col_input, col_example = st.columns([2, 1])
    with col_input:
        bbl_input = st.text_input("Enter BBL", placeholder="e.g. 1000750043", max_chars=15)
    with col_example:
        st.markdown("**Example BBLs**")
        for b in df.sample(5, random_state=42)["BBL"].tolist():
            if st.button(b, key=f"btn_{b}"):
                bbl_input = b

    if bbl_input:
        prop = lookup_bbl(bbl_input, df)

        if prop is None:
            st.error(f"BBL `{bbl_input}` not found. Please check the number.")
        else:
            st.subheader(f"Property: BBL {bbl_input}")

            # ── Property info ─────────────────────────────────────────────────
            r1 = st.columns(4)
            r1[0].metric("Borough",       BORO_MAP.get(str(prop.get("BORO", "")), "Unknown"))
            r1[1].metric("Building Class", str(prop.get("BLDG_CLASS", "N/A")))
            r1[2].metric("Gross Sqft",     f"{float(prop.get('GROSS_SQFT', 0) or 0):,.0f}")
            r1[3].metric("Year Built",     str(prop.get("YRBUILT", "N/A")))
            r2 = st.columns(4)
            r2[0].metric("ZIP Code",       str(prop.get("ZIP_CODE", "N/A")).strip())
            r2[1].metric("Units",          str(prop.get("UNITS",    "N/A")))
            r2[2].metric("# Buildings",    str(prop.get("NUM_BLDGS","N/A")))
            r2[3].metric("Floors",         str(prop.get("BLD_STORY","N/A")))

            st.markdown("---")

            # ── Classification + history chart ────────────────────────────────
            actual = str(prop.get("target_2026", "unknown"))
            color  = CLASS_COLORS.get(actual, "#999")
            label  = CLASS_LABELS.get(actual, actual)

            pred_col, hist_col = st.columns([1, 1])

            with pred_col:
                st.subheader("FY2026 Classification")
                st.markdown(
                    f"<div style='background:{color}22; border-left:6px solid {color};"
                    f"padding:16px; border-radius:8px; font-size:1.4em; font-weight:bold;'>"
                    f"{label}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.markdown("""
                Compares this property's **assessed value per square foot**
                to the median of all properties in the same **borough + building class**.

                - 🔴 >15% above peer median
                - 🟢 Within ±15% of peer median
                - 🔵 >15% below peer median
                """)

            with hist_col:
                st.subheader("Assessment History (FY2020–FY2026)")
                hist_years = [2020, 2021, 2022, 2023, 2024, 2025]
                hist_vals  = [pd.to_numeric(prop.get(f"FINACTTOT_FY{y}", np.nan), errors="coerce") for y in hist_years]
                hist_years.append(2026)
                hist_vals.append(pd.to_numeric(prop.get("FINACTTOT", np.nan), errors="coerce"))

                valid = [(y, v) for y, v in zip(hist_years, hist_vals) if pd.notna(v)]
                if valid:
                    ys, vs = zip(*valid)
                    fig4, ax4 = plt.subplots(figsize=(6, 3))
                    ax4.plot(ys, vs, marker="o", color="#4575b4", linewidth=2)
                    ax4.fill_between(ys, vs, alpha=0.1, color="#4575b4")
                    ax4.set_xlabel("Fiscal Year")
                    ax4.set_ylabel("Assessed Value ($)")
                    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close()
                else:
                    st.info("No historical assessment data available.")

            # ── Peer group comparison ─────────────────────────────────────────
            st.markdown("---")
            st.subheader("Peer Group Comparison")

            boro_val   = prop.get("BORO", "")
            bldg_val   = prop.get("BLDG_CLASS", "")
            gross_sqft = pd.to_numeric(prop.get("GROSS_SQFT", 0), errors="coerce")
            finacttot  = pd.to_numeric(prop.get("FINACTTOT",  np.nan), errors="coerce")

            peers = df[(df["BORO"] == boro_val) & (df["BLDG_CLASS"] == bldg_val)].copy()

            if len(peers) > 0 and gross_sqft and gross_sqft > 0 and pd.notna(finacttot):
                peers["assess_per_sqft"] = peers["FINACTTOT"] / peers["GROSS_SQFT"].clip(lower=1)
                this_psqft  = finacttot / gross_sqft
                peer_median = peers["assess_per_sqft"].median()
                peer_p25    = peers["assess_per_sqft"].quantile(0.25)
                peer_p75    = peers["assess_per_sqft"].quantile(0.75)

                pc = st.columns(4)
                pc[0].metric("This Property ($/sqft)", f"${this_psqft:,.2f}")
                pc[1].metric("Peer Median ($/sqft)",   f"${peer_median:,.2f}")
                pc[2].metric("vs Peer Median",          f"{(this_psqft/peer_median - 1)*100:+.1f}%")
                pc[3].metric("Peer Group Size",          f"{len(peers):,}")

                plot_data = peers["assess_per_sqft"].dropna()
                plot_data = plot_data[plot_data.between(plot_data.quantile(0.01), plot_data.quantile(0.99))]
                fig5, ax5 = plt.subplots(figsize=(8, 3))
                ax5.hist(plot_data, bins=50, color="#aaaaaa", alpha=0.7, label="Peer group")
                ax5.axvline(this_psqft,  color=color,    linewidth=3, label=f"This: ${this_psqft:,.0f}")
                ax5.axvline(peer_median, color="#333333", linewidth=2, linestyle="--", label=f"Median: ${peer_median:,.0f}")
                ax5.axvspan(peer_median * 0.85, peer_median * 1.15, alpha=0.1, color="green", label="±15% fair zone")
                ax5.set_xlabel("Assessed Value per Sqft ($)")
                ax5.set_ylabel("Count")
                ax5.set_title(f"Peer Group — {BORO_MAP.get(str(boro_val), '')} / {bldg_val}")
                ax5.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig5)
                plt.close()
                st.caption(f"25th–75th percentile: ${peer_p25:,.0f}–${peer_p75:,.0f}/sqft")
            else:
                st.info("Not enough peer data to show comparison.")

            # ── Historical classification ─────────────────────────────────────
            st.markdown("---")
            st.subheader("Historical Classification (FY2020–FY2025)")
            hist_cols = st.columns(6)
            for i, yr in enumerate([2020, 2021, 2022, 2023, 2024, 2025]):
                over  = int(prop.get(f"overvalued_{yr}",    0) or 0)
                under = int(prop.get(f"undervalued_{yr}",   0) or 0)
                if over:
                    s, c = "Overvalued",    "#d73027"
                elif under:
                    s, c = "Undervalued",   "#4575b4"
                else:
                    s, c = "Fairly Valued", "#1a9850"
                hist_cols[i].markdown(
                    f"<div style='text-align:center;padding:8px;border-radius:6px;"
                    f"background:{c}22;border:2px solid {c}'>"
                    f"<b>FY{yr}</b><br><span style='color:{c}'>{s}</span></div>",
                    unsafe_allow_html=True,
                )