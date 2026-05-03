"""
app.py
======
Streamlit app for NYC Property Tax Assessment Classification
Three tabs:
  1. Methodology — model explanation, feature importance, confusion matrix
  2. Borough Analysis — maps and charts of assessment patterns
  3. BBL Lookup — enter a BBL and get a prediction with explanation
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import sys

# ── Add python_scripts to path so feature_engineering imports work ────────────
sys.path.insert(0, "/home/rodrigofrancachaves/project-nyc_property_taxes/python_scripts")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "/home/rodrigofrancachaves/project-nyc_property_taxes/data/processed_labeled_data.parquet"
MODEL_DIR    = "/home/rodrigofrancachaves/project-nyc_property_taxes/models"
OUTPUT_DIR   = "/home/rodrigofrancachaves/project-nyc_property_taxes/outputs"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Property Tax Assessment",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color map ─────────────────────────────────────────────────────────────────
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
BORO_MAP = {
    "1": "Manhattan",
    "2": "Bronx",
    "3": "Brooklyn",
    "4": "Queens",
    "5": "Staten Island",
}


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model    = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    le_dict  = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    return model, features, le_dict


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["BBL"] = df["BBL"].astype(str).str.strip()
    # Borough name
    df["BORO_NAME"] = df["BORO"].astype(str).map(BORO_MAP).fillna("Unknown")
    return df


@st.cache_data(show_spinner="Loading feature importance...")
def load_feature_importance():
    path = os.path.join(OUTPUT_DIR, "lgbm_feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_bbl(bbl: str, df, model, features, le_dict):
    """
    Look up BBL in the dataset, run feature engineering,
    and return prediction + probability + feature contributions.
    """
    row = df[df["BBL"] == bbl.strip()]
    if row.empty:
        return None, None, None, None

    # Import engineer_features to apply same pipeline
    from feature_engineering import engineer_features
    row_eng, feat_list, _ = engineer_features(row.copy(), le_dict=le_dict)

    # Keep only features the model knows about
    feat_list = [f for f in features if f in row_eng.columns]
    X = row_eng[feat_list].copy()
    for col in feat_list:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    pred      = model.predict(X)[0]
    proba     = model.predict_proba(X)[0]
    classes   = model.classes_

    proba_dict = dict(zip(classes, proba))
    return pred, proba_dict, row.iloc[0], feat_list


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/NYC_wikilogo.svg/200px-NYC_wikilogo.svg.png", width=80)
    st.title("NYC Property Tax")
    st.markdown("**Assessment Classification Model**")
    st.markdown("---")
    st.markdown("""
    This tool uses a **LightGBM** model trained on NYC Department of Finance
    assessment data (FY2020–FY2026) to classify properties as:

    - 🔴 **Overvalued** — assessed >15% above peer median
    - 🟢 **Fairly Valued** — within ±15% of peer median
    - 🔵 **Undervalued** — assessed >15% below peer median

    Peer groups are defined by **borough + building class**.
    """)
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.metric("Test F1 Macro", "90.0%")
    st.metric("Test Accuracy", "89.5%")
    st.metric("Training Properties", "1.1M")
    st.markdown("---")
    st.caption("CAPP 30254 · Spring 2026 · UChicago")


# ── Load everything ───────────────────────────────────────────────────────────
model, features, le_dict = load_model()
df = load_data()
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
        based on their assessed value per square foot relative to peer properties
        (same borough + building class), using structural, geographic, and
        historical assessment features?
        """)

        st.subheader("Data")
        st.markdown("""
        - **Source:** NYC Department of Finance Property Assessment Rolls
        - **Years:** FY2020–FY2026
        - **Properties:** ~1.1 million tax lots
        - **Target:** FY2026 peer-group classification (±15% threshold)
        """)

        st.subheader("Labeling Strategy")
        st.markdown("""
        Each property is labeled by comparing its **assessed value per square foot**
        to the **median of its peer group** (same borough + building class):

        | Label | Condition |
        |-------|-----------|
        | Overvalued | > 115% of peer median |
        | Fairly Valued | 85–115% of peer median |
        | Undervalued | < 85% of peer median |

        Historical labels (FY2020–FY2025) become **features** for predicting FY2026.
        """)

        st.subheader("Model")
        st.markdown("""
        **LightGBM** (gradient boosting, leaf-wise tree growth)
        - Trained on 300k stratified subsample
        - Hyperparameters tuned via cross-validated random search
        - `class_weight="balanced"` to handle class imbalance
        - **Primary metric:** Macro F1 (equal weight per class)
        """)

    with col2:
        st.subheader("Model Performance")

        # Performance table
        perf_df = pd.DataFrame({
            "Model":            ["SGD L2", "SGD L1", "SGD ElasticNet", "Passive Aggressive", "LightGBM"],
            "Type":             ["Linear", "Linear", "Linear", "Linear", "Non-linear"],
            "Test F1 Macro":    [0.864, 0.877, 0.878, 0.842, 0.900],
            "Test Accuracy":    [0.861, 0.874, 0.875, 0.838, 0.895],
            "CV F1 Macro":      [0.860, 0.874, 0.875, 0.843, 0.897],
        })
        st.dataframe(
            perf_df.style.highlight_max(
                subset=["Test F1 Macro", "Test Accuracy", "CV F1 Macro"],
                color="#c6efce"
            ).format({
                "Test F1 Macro": "{:.3f}",
                "Test Accuracy": "{:.3f}",
                "CV F1 Macro":   "{:.3f}",
            }),
            use_container_width=True,
        )
        st.caption("Baseline (majority class): 0.427")

        # Feature importance plot
        st.subheader("Top Features (LightGBM)")
        if feat_imp is not None:
            fig, ax = plt.subplots(figsize=(8, 7))
            top20 = feat_imp.head(20)
            ax.barh(top20["Feature"][::-1], top20["Importance"][::-1], color="#4575b4")
            ax.set_xlabel("Importance")
            ax.set_title("Top 20 Feature Importances")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Run the model first to generate feature importance.")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm_path = os.path.join(OUTPUT_DIR, "lgbm_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            st.info("Run the model first to generate the confusion matrix.")

    st.subheader("Feature Groups")
    feat_groups = {
        "📍 Location":           ["BORO_CODE", "ZIP_CODE_CODE", "ZONING_CODE", "BLDG_CLASS_CODE"],
        "🏗️ Physical":           ["LOG_GROSS_SQFT", "LOG_LAND_AREA", "BUILDING_AGE", "SQFT_PER_UNIT", "COVERAGE_RATIO"],
        "💰 Assessment (FY2025)":["ASSESS_PER_SQFT", "LAND_TO_TOTAL", "MKT_TO_ASSESS"],
        "📈 Trend & Momentum":   ["ASSESS_TREND", "ASSESS_VOLATILITY", "ASSESS_AT_CAP"],
        "🔄 YoY % Changes":      ["ASSESS_YOY_FY2021–2025", "LAND_YOY_FY2021–2025", "MKT_YOY_FY2021–2025"],
        "📊 Historical Status":  ["overvalued_2020–2025", "undervalued_2020–2025", "fairly_valued_2020–2025"],
        "🔮 OLS Projections":    ["PROJ_FINACTTOT_FY2026", "PROJ_RATIO_FINACTTOT_FY2026"],
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

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_boros = st.multiselect(
            "Filter by Borough",
            options=list(BORO_MAP.values()),
            default=list(BORO_MAP.values()),
        )
    with col_f2:
        selected_classes = st.multiselect(
            "Filter by Predicted Class",
            options=["overvalued", "fairly_valued", "undervalued"],
            default=["overvalued", "fairly_valued", "undervalued"],
            format_func=lambda x: CLASS_LABELS[x],
        )

    df_filtered = df[
        df["BORO_NAME"].isin(selected_boros) &
        df["target_2026"].isin(selected_classes)
    ]

    st.caption(f"Showing {len(df_filtered):,} properties")

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    vc = df_filtered["target_2026"].value_counts()
    m1.metric("🔴 Overvalued",    f"{vc.get('overvalued',    0):,}", f"{vc.get('overvalued',    0)/len(df_filtered)*100:.1f}%")
    m2.metric("🟢 Fairly Valued", f"{vc.get('fairly_valued', 0):,}", f"{vc.get('fairly_valued', 0)/len(df_filtered)*100:.1f}%")
    m3.metric("🔵 Undervalued",   f"{vc.get('undervalued',   0):,}", f"{vc.get('undervalued',   0)/len(df_filtered)*100:.1f}%")
    m4.metric("📦 Total",         f"{len(df_filtered):,}")

    # ── Charts ────────────────────────────────────────────────────────────────
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("Classification by Borough")
        boro_class = (
            df_filtered.groupby(["BORO_NAME", "target_2026"])
            .size().reset_index(name="count")
        )
        boro_pivot = boro_class.pivot(
            index="BORO_NAME", columns="target_2026", values="count"
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(8, 5))
        boro_pivot.plot(
            kind="bar", ax=ax, stacked=True,
            color=[CLASS_COLORS.get(c, "#999") for c in boro_pivot.columns],
        )
        ax.set_xlabel("")
        ax.set_ylabel("Number of Properties")
        ax.set_title("Property Classification by Borough")
        ax.legend(title="Classification", bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_c2:
        st.subheader("Classification by Building Class (Top 15)")
        bldg_class = (
            df_filtered.groupby(["BLDG_CLASS", "target_2026"])
            .size().reset_index(name="count")
        )
        # Top 15 building classes by count
        top_bldg = (
            df_filtered["BLDG_CLASS"].value_counts().head(15).index.tolist()
        )
        bldg_pivot = (
            bldg_class[bldg_class["BLDG_CLASS"].isin(top_bldg)]
            .pivot(index="BLDG_CLASS", columns="target_2026", values="count")
            .fillna(0)
        )
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        bldg_pivot.plot(
            kind="bar", ax=ax2, stacked=True,
            color=[CLASS_COLORS.get(c, "#999") for c in bldg_pivot.columns],
        )
        ax2.set_xlabel("")
        ax2.set_ylabel("Number of Properties")
        ax2.set_title("Top 15 Building Classes")
        ax2.legend(title="Classification", bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Assessment per sqft by borough ────────────────────────────────────────
    st.subheader("Assessed Value per Sqft Distribution by Borough")
    if "FINACTTOT" in df_filtered.columns and "GROSS_SQFT" in df_filtered.columns:
        df_filtered = df_filtered.copy()
        df_filtered["assess_per_sqft"] = (
            pd.to_numeric(df_filtered["FINACTTOT"], errors="coerce") /
            pd.to_numeric(df_filtered["GROSS_SQFT"], errors="coerce").clip(lower=1)
        )
        df_plot = df_filtered[df_filtered["assess_per_sqft"].between(0, 500)]

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        for boro in selected_boros:
            data = df_plot[df_plot["BORO_NAME"] == boro]["assess_per_sqft"].dropna()
            if not data.empty:
                ax3.hist(data, bins=60, alpha=0.5, label=boro, density=True)
        ax3.set_xlabel("Assessed Value per Sqft ($)")
        ax3.set_ylabel("Density")
        ax3.set_title("Distribution of Assessed Value per Sqft")
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # ── Data table ────────────────────────────────────────────────────────────
    st.subheader("Summary Table by Borough")
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
    summary["% Overvalued"]  = (summary["Overvalued"]    / summary["Total"] * 100).round(1)
    summary["% Fairly Valued"] = (summary["Fairly_Valued"] / summary["Total"] * 100).round(1)
    summary["% Undervalued"] = (summary["Undervalued"]   / summary["Total"] * 100).round(1)
    st.dataframe(summary, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — BBL LOOKUP
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🔍 Property Lookup")
    st.markdown("""
    Enter a **BBL (Borough-Block-Lot)** number to get a classification prediction
    and see how this property compares to its peers.

    BBL format: `BBBBBBBLLLL` (10 digits) — e.g. `1000750043` for a Manhattan property.
    """)

    col_input, col_example = st.columns([2, 1])
    with col_input:
        bbl_input = st.text_input(
            "Enter BBL",
            placeholder="e.g. 1000750043",
            max_chars=15,
        )
    with col_example:
        st.markdown("**Example BBLs**")
        example_bbls = df.sample(5, random_state=42)["BBL"].tolist()
        for b in example_bbls:
            if st.button(b, key=f"btn_{b}"):
                bbl_input = b

    if bbl_input:
        bbl_input = bbl_input.strip()

        # Look up in dataset
        row = df[df["BBL"] == bbl_input]

        if row.empty:
            st.error(f"BBL `{bbl_input}` not found in dataset. Please check the number.")
        else:
            prop = row.iloc[0]

            # ── Property info ─────────────────────────────────────────────────
            st.subheader(f"Property: BBL {bbl_input}")

            info_cols = st.columns(4)
            info_cols[0].metric("Borough",        BORO_MAP.get(str(prop.get("BORO", "")), "Unknown"))
            info_cols[1].metric("Building Class",  str(prop.get("BLDG_CLASS", "N/A")))
            info_cols[2].metric("Gross Sqft",      f"{float(prop.get('GROSS_SQFT', 0)):,.0f}")
            info_cols[3].metric("Year Built",      str(prop.get("YRBUILT", "N/A")))

            info_cols2 = st.columns(4)
            info_cols2[0].metric("ZIP Code",       str(prop.get("ZIP_CODE", "N/A")).strip())
            info_cols2[1].metric("Units",          str(prop.get("UNITS", "N/A")))
            info_cols2[2].metric("# Buildings",    str(prop.get("NUM_BLDGS", "N/A")))
            info_cols2[3].metric("Floors",         str(prop.get("BLD_STORY", "N/A")))

            st.markdown("---")

            # ── Prediction ────────────────────────────────────────────────────
            actual = str(prop.get("target_2026", "unknown"))
            color  = CLASS_COLORS.get(actual, "#999")
            label  = CLASS_LABELS.get(actual,  actual)

            pred_col, assess_col = st.columns([1, 1])

            with pred_col:
                st.subheader("Classification")
                st.markdown(
                    f"<div style='background-color:{color}22; border-left: 6px solid {color};"
                    f"padding: 16px; border-radius: 8px; font-size: 1.4em; font-weight: bold;'>"
                    f"{label}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.markdown("""
                **What does this mean?**

                This classification compares the property's **assessed value per square foot**
                to the median of all properties in the same **borough + building class** group.

                - 🔴 **Overvalued**: assessed more than 15% above peer median
                - 🟢 **Fairly Valued**: within ±15% of peer median
                - 🔵 **Undervalued**: assessed more than 15% below peer median
                """)

            with assess_col:
                st.subheader("Assessment History")

                # Build history chart
                hist_years = [2020, 2021, 2022, 2023, 2024, 2025]
                hist_vals  = []
                for yr in hist_years:
                    col_name = f"FINACTTOT_FY{yr}"
                    val = pd.to_numeric(prop.get(col_name, np.nan), errors="coerce")
                    hist_vals.append(val)

                # Add FY2026
                val_2026 = pd.to_numeric(prop.get("FINACTTOT", np.nan), errors="coerce")
                hist_years.append(2026)
                hist_vals.append(val_2026)

                if any(pd.notna(v) for v in hist_vals):
                    fig4, ax4 = plt.subplots(figsize=(6, 3))
                    valid = [(y, v) for y, v in zip(hist_years, hist_vals) if pd.notna(v)]
                    ys, vs = zip(*valid)
                    ax4.plot(ys, vs, marker="o", color="#4575b4", linewidth=2)
                    ax4.fill_between(ys, vs, alpha=0.1, color="#4575b4")
                    ax4.set_xlabel("Fiscal Year")
                    ax4.set_ylabel("Assessed Value ($)")
                    ax4.set_title("Total Assessed Value FY2020–FY2026")
                    ax4.yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
                    )
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close()
                else:
                    st.info("No historical assessment data available.")

            # ── Peer group comparison ─────────────────────────────────────────
            st.markdown("---")
            st.subheader("Peer Group Comparison")

            boro_val  = prop.get("BORO", "")
            bldg_val  = prop.get("BLDG_CLASS", "")
            gross_sqft = pd.to_numeric(prop.get("GROSS_SQFT", 0), errors="coerce")
            finacttot  = pd.to_numeric(prop.get("FINACTTOT", np.nan), errors="coerce")

            peers = df[
                (df["BORO"] == boro_val) &
                (df["BLDG_CLASS"] == bldg_val)
            ]

            if len(peers) > 0 and gross_sqft > 0 and pd.notna(finacttot):
                peers = peers.copy()
                peers["assess_per_sqft"] = (
                    pd.to_numeric(peers["FINACTTOT"], errors="coerce") /
                    pd.to_numeric(peers["GROSS_SQFT"], errors="coerce").clip(lower=1)
                )
                this_psqft    = finacttot / gross_sqft
                peer_median   = peers["assess_per_sqft"].median()
                peer_p25      = peers["assess_per_sqft"].quantile(0.25)
                peer_p75      = peers["assess_per_sqft"].quantile(0.75)

                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("This Property ($/sqft)",  f"${this_psqft:,.2f}")
                pc2.metric("Peer Median ($/sqft)",    f"${peer_median:,.2f}")
                pc3.metric("vs Peer Median",           f"{(this_psqft/peer_median - 1)*100:+.1f}%")
                pc4.metric("Peer Group Size",          f"{len(peers):,} properties")

                # Distribution chart
                fig5, ax5 = plt.subplots(figsize=(8, 3))
                plot_data = peers["assess_per_sqft"].dropna()
                plot_data = plot_data[plot_data.between(
                    plot_data.quantile(0.01),
                    plot_data.quantile(0.99)
                )]
                ax5.hist(plot_data, bins=50, color="#aaaaaa", alpha=0.7, label="Peer group")
                ax5.axvline(this_psqft,  color=color,     linewidth=3, label=f"This property: ${this_psqft:,.0f}")
                ax5.axvline(peer_median, color="#333333",  linewidth=2, linestyle="--", label=f"Peer median: ${peer_median:,.0f}")
                ax5.axvspan(peer_median * 0.85, peer_median * 1.15, alpha=0.1, color="green", label="±15% fair zone")
                ax5.set_xlabel("Assessed Value per Sqft ($)")
                ax5.set_ylabel("Count")
                ax5.set_title(f"Peer Group Distribution — {BORO_MAP.get(str(boro_val), '')} / {bldg_val}")
                ax5.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig5)
                plt.close()

                st.caption(
                    f"Peer group: {BORO_MAP.get(str(boro_val), 'Unknown')} borough, "
                    f"building class {bldg_val} ({len(peers):,} properties). "
                    f"25th–75th percentile: ${peer_p25:,.0f}–${peer_p75:,.0f}/sqft."
                )
            else:
                st.info("Not enough peer data to show comparison.")

            # ── Historical status ─────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Historical Classification (FY2020–FY2025)")

            hist_status_cols = st.columns(6)
            for i, yr in enumerate([2020, 2021, 2022, 2023, 2024, 2025]):
                over  = int(prop.get(f"overvalued_{yr}",    0))
                under = int(prop.get(f"undervalued_{yr}",   0))
                fair  = int(prop.get(f"fairly_valued_{yr}", 0))
                if over:
                    status, clr = "Overvalued",    "#d73027"
                elif under:
                    status, clr = "Undervalued",   "#4575b4"
                else:
                    status, clr = "Fairly Valued", "#1a9850"
                hist_status_cols[i].markdown(
                    f"<div style='text-align:center; padding:8px; border-radius:6px;"
                    f"background:{clr}22; border:2px solid {clr}'>"
                    f"<b>FY{yr}</b><br><span style='color:{clr}'>{status}</span></div>",
                    unsafe_allow_html=True,
                )