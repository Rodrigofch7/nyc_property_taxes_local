"""
app.py
======
Streamlit app for NYC Property Tax Assessment Classification
Three tabs:
  1. Methodology — model explanation, feature importance, confusion matrix
  2. Borough Analysis — pre-aggregated charts (no parquet)
  3. BBL Lookup — 100-property demo sample (no parquet)

No full dataset loaded — all data served from pre-aggregated JSONs.
Deploy: streamlit run app_deploy/app.py
"""

import os
import json
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))  # app_deploy/
PROJECT_DIR = os.path.dirname(BASE_DIR)                   # repo root
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs")

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


PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def style_fig(fig, height=380, title=None):
    """Apply consistent, lightweight styling to a Plotly figure."""
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#333333", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
    return fig


def stacked_class_bar(pivot_df, title, tickangle=-30):
    """Stacked bar of property counts by class over a pivoted index."""
    fig = go.Figure()
    for cls in ["undervalued", "fairly_valued", "overvalued"]:
        if cls in pivot_df.columns:
            fig.add_bar(
                x=pivot_df.index, y=pivot_df[cls],
                name=CLASS_LABELS.get(cls, cls),
                marker_color=CLASS_COLORS.get(cls),
                hovertemplate="%{x}<br>" + CLASS_LABELS.get(cls, cls) + ": %{y:,}<extra></extra>",
            )
    fig.update_layout(barmode="stack", xaxis_title="", yaxis_title="Properties", legend_title="Class")
    fig.update_xaxes(tickangle=tickangle)
    return style_fig(fig, title=title)


# ── Custom styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: #f8f9fb;
    border: 1px solid #e6e9ef;
    border-radius: 10px;
    padding: 14px 16px 10px 16px;
}
div[data-testid="stMetricValue"] { font-size: 1.55rem; }
div[data-testid="stMetricLabel"] { font-weight: 600; color: #555; }
button[data-baseweb="tab"] p { font-size: 1.02rem; font-weight: 600; }
h2, h3 { letter-spacing: -0.01em; }
section[data-testid="stSidebar"] { border-right: 1px solid #e6e9ef; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model    = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    le_dict  = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    return model, features, le_dict


@st.cache_data(show_spinner=False)
def load_feature_importance():
    path = os.path.join(OUTPUT_DIR, "lgbm_feature_importance.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def load_borough_summary():
    with open(os.path.join(BASE_DIR, "borough_summary.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_sample_properties():
    with open(os.path.join(BASE_DIR, "sample_properties.json")) as f:
        data = json.load(f)
    return data, {str(p["BBL"]): p for p in data}


# ── Load everything ───────────────────────────────────────────────────────────
model, features, le_dict   = load_model()
feat_imp                   = load_feature_importance()
bsummary                   = load_borough_summary()
sample_list, sample_lookup = load_sample_properties()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists(icon_path):
        st.image(icon_path, width=80)
    st.title("NYC Property Tax")
    st.caption("Assessment Classification Model")
    st.divider()
    st.markdown("""
    **LightGBM** trained on NYC DOF assessment data (FY2020–FY2026)
    to classify ~1.1M properties as:

    - 🔴 **Overvalued** — assessed >15% above peer median
    - 🟢 **Fairly Valued** — within ±15% of peer median
    - 🔵 **Undervalued** — assessed >15% below peer median

    Peer groups: borough + building class (up to 6-level fallback hierarchy).
    """)
    st.divider()
    st.caption("CAPP 30254 · Spring 2026 · UChicago  \nAhmed Lodhi · Faizan Imran · Rodrigo Chaves")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background: linear-gradient(100deg, #0b3d91 0%, #1a5fb4 100%);
                padding: 28px 32px; border-radius: 14px; margin-bottom: 16px;">
        <h1 style="color:white; margin:0; font-size:2rem;">🏙️ NYC Property Tax Assessment Classifier</h1>
        <p style="color:#dbe9ff; margin:8px 0 0 0; font-size:1.05rem; max-width:720px;">
            A LightGBM model flags NYC properties as over-, under-, or fairly assessed
            relative to their peers — trained on 1.1M tax lots and six years of DOF assessment history.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

hero1, hero2, hero3, hero4 = st.columns(4)
hero1.metric("Test F1 Macro", "86.7%")
hero2.metric("Test Accuracy", "87.6%")
hero3.metric("Properties Trained On", "877k")
hero4.metric("Properties Classified", f"{bsummary['totals']['total']:,}")

st.write("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Methodology", "🗺️ Borough Analysis", "🔍 BBL Lookup"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — METHODOLOGY
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Methodology")

    st.info(
        "**Key result:** LightGBM reaches **86.7% Macro F1** — a **+6.5 point** gain "
        "over the best linear baseline (SGD L2, 80.3%) and **+29.7 points** over the "
        "majority-class baseline (57.0%), driven mainly by peer-relative assessment "
        "ratios and multi-year assessment trend features."
    )

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
            "Test F1 Macro":  [0.8027, 0.8024, 0.7733, 0.8672],
            "Test Accuracy":  [0.8149, 0.8143, 0.7909, 0.8761],
            "CV F1 Macro":    [0.8010, 0.8006, 0.7456, None],
        })
        def highlight_best(s):
            is_max = s == s.max()
            return ["color: #f5c518; font-weight: bold" if v else "" for v in is_max]

        st.dataframe(
            perf_df.style
            .apply(highlight_best, subset=["Test F1 Macro", "Test Accuracy"])
            .format({
                "Test F1 Macro": "{:.4f}",
                "Test Accuracy": "{:.4f}",
                "CV F1 Macro":   lambda x: f"{x:.4f}" if pd.notna(x) else "—",
            }),
            width='stretch',
        )
        st.caption("Baseline (majority class): 0.570  |  Linear models: PCA (60 components) + StandardScaler")

        st.subheader("Top Features (LightGBM)")
        if feat_imp is not None:
            top20 = feat_imp.head(20).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top20["Importance"], y=top20["Feature"], orientation="h",
                marker_color="#4575b4",
                hovertemplate="%{y}<br>Importance: %{x:.1f}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Importance", yaxis_title="")
            style_fig(fig, height=520, title="Top 20 Feature Importances")
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
        else:
            st.info("Feature importance CSV not found in outputs/.")

        cm_path = os.path.join(OUTPUT_DIR, "lgbm_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.subheader("Confusion Matrix")
            st.image(cm_path, width='stretch')

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
# TAB 2 — BOROUGH ANALYSIS (pre-aggregated JSON — no parquet)
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Borough Analysis")
    st.caption("Based on full dataset of 1,110,445 NYC tax lots (FY2026)")

    totals = bsummary["totals"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔴 Overvalued",    f"{totals['overvalued']:,}",    f"{totals['overvalued']/totals['total']*100:.1f}%")
    m2.metric("🟢 Fairly Valued", f"{totals['fairly_valued']:,}", f"{totals['fairly_valued']/totals['total']*100:.1f}%")
    m3.metric("🔵 Undervalued",   f"{totals['undervalued']:,}",   f"{totals['undervalued']/totals['total']*100:.1f}%")
    m4.metric("📦 Total",         f"{totals['total']:,}")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("Classification by Borough")
        boro_df    = pd.DataFrame(bsummary["boro_class"])
        boro_df    = boro_df[boro_df["target_2026"] != "unknown"]
        boro_pivot = boro_df.pivot(index="BORO_NAME", columns="target_2026", values="count").fillna(0)

        fig = stacked_class_bar(boro_pivot, "Classification by Borough", tickangle=-30)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    with col_c2:
        st.subheader("Top 15 Building Classes")
        bldg_df    = pd.DataFrame(bsummary["bldg_class"])
        bldg_df    = bldg_df[bldg_df["target_2026"] != "unknown"]
        bldg_pivot = bldg_df.pivot(index="BLDG_CLASS", columns="target_2026", values="count").fillna(0)

        fig2 = stacked_class_bar(bldg_pivot, "Top 15 Building Classes", tickangle=-45)
        st.plotly_chart(fig2, width='stretch', config=PLOTLY_CONFIG)

    st.subheader("Assessed Value per Sqft Distribution by Borough")
    selected_boros = st.multiselect(
        "Select boroughs",
        options=list(bsummary["hist_data"].keys()),
        default=list(bsummary["hist_data"].keys()),
    )
    fig3 = go.Figure()
    for boro in selected_boros:
        hdata   = bsummary["hist_data"][boro]
        counts  = np.array(hdata["counts"])
        edges   = np.array(hdata["edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        total   = counts.sum()
        fig3.add_scatter(x=centers, y=counts / total, mode="lines", name=boro,
                          line=dict(width=2), hovertemplate="$%{x:.0f}<br>Density: %{y:.4f}<extra>%{fullData.name}</extra>")
    fig3.update_layout(xaxis_title="Assessed Value per Sqft ($)", yaxis_title="Density")
    fig3.update_xaxes(range=[0, 300])
    style_fig(fig3, height=340, title="Distribution of Assessed Value per Sqft")
    st.plotly_chart(fig3, width='stretch', config=PLOTLY_CONFIG)

    st.subheader("Summary by Borough")
    summary_df = pd.DataFrame(bsummary["summary"]).rename(columns={
        "BORO_NAME": "Borough", "Fairly_Valued": "Fairly Valued",
        "pct_over": "% Over", "pct_fair": "% Fair", "pct_under": "% Under",
    })
    st.dataframe(summary_df, width='stretch')


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — BBL LOOKUP (sample_properties.json — no parquet)
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🔍 Property Lookup")
    st.info(
        "🔎 **Demo sample** — 100 real properties, stratified across all 3 classes and "
        "5 boroughs. Browse or search the full list below, or enter a BBL directly. "
        "The full model was trained on 1.1M NYC tax lots."
    )

    if "bbl_text" not in st.session_state:
        st.session_state.bbl_text = ""

    def _set_bbl(value):
        st.session_state.bbl_text = value

    def _sample_label(p):
        boro = BORO_MAP.get(str(p.get("BORO", "")), "")
        cls  = CLASS_LABELS.get(p.get("target_2026", ""), "")
        return f"{p['BBL']} — {boro}, {p.get('BLDG_CLASS', '')} ({cls})"

    sample_options = {_sample_label(p): str(p["BBL"]) for p in sample_list}

    def _apply_pick():
        label = st.session_state.get("bbl_dropdown", "")
        if label:
            st.session_state.bbl_text = sample_options[label]

    col_input, col_pick = st.columns([1, 2])
    with col_input:
        st.text_input("Enter a BBL directly", key="bbl_text", placeholder="e.g. 5036410049", max_chars=15)
    with col_pick:
        st.selectbox(
            "…or search/browse all 100 sample properties",
            options=[""] + sorted(sample_options.keys()),
            key="bbl_dropdown",
            on_change=_apply_pick,
        )

    shown, quick = set(), []
    for p in sample_list:
        cls = p.get("target_2026", "")
        if cls not in shown and cls in CLASS_LABELS:
            quick.append(p)
            shown.add(cls)
        if len(shown) == 3:
            break
    st.caption("Quick picks — one per class:")
    qcols = st.columns(3)
    for i, p in enumerate(quick):
        b   = str(p["BBL"])
        lbl = CLASS_LABELS.get(p.get("target_2026", ""), b)
        qcols[i].button(f"{b} ({lbl})", key=f"btn_{b}", on_click=_set_bbl, args=(b,))

    bbl_input = st.session_state.bbl_text

    if bbl_input:
        prop = sample_lookup.get(bbl_input.strip())

        if prop is None:
            st.error(f"BBL `{bbl_input}` not in the demo sample. Try one of the example BBLs above.")
        else:
            st.subheader(f"Property: BBL {bbl_input}")

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

            st.divider()

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
                hist_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
                hist_vals  = [
                    prop.get(f"FINACTTOT_FY{y}", None) for y in [2020, 2021, 2022, 2023, 2024, 2025]
                ] + [prop.get("FINACTTOT", None)]

                valid = [(y, float(v)) for y, v in zip(hist_years, hist_vals) if v not in (None, 0)]
                if valid:
                    ys, vs = zip(*valid)
                    fig4 = go.Figure(go.Scatter(
                        x=ys, y=vs, mode="lines+markers",
                        line=dict(color="#4575b4", width=2), marker=dict(size=7),
                        fill="tozeroy", fillcolor="rgba(69,117,180,0.12)",
                        hovertemplate="FY%{x}: $%{y:,.0f}<extra></extra>",
                    ))
                    fig4.update_layout(xaxis_title="Fiscal Year", yaxis_title="Assessed Value ($)")
                    fig4.update_xaxes(dtick=1)
                    fig4.update_yaxes(tickprefix="$", separatethousands=True)
                    style_fig(fig4, height=300)
                    st.plotly_chart(fig4, width='stretch', config=PLOTLY_CONFIG)
                else:
                    st.info("No historical assessment data available.")

            st.divider()
            st.subheader("Peer Group Comparison")

            gross_sqft  = float(prop.get("GROSS_SQFT", 0) or 0)
            finacttot   = float(prop.get("FINACTTOT",  0) or 0)
            peer_median = float(prop.get("peer_median", 0) or 0)
            peer_p25    = float(prop.get("peer_p25",    0) or 0)
            peer_p75    = float(prop.get("peer_p75",    0) or 0)
            peer_size   = int(prop.get("peer_size",     0) or 0)

            if gross_sqft > 0 and finacttot > 0 and peer_median > 0:
                this_psqft = finacttot / gross_sqft

                pc = st.columns(4)
                pc[0].metric("This Property ($/sqft)", f"${this_psqft:,.2f}")
                pc[1].metric("Peer Median ($/sqft)",   f"${peer_median:,.2f}")
                pc[2].metric("vs Peer Median",          f"{(this_psqft/peer_median - 1)*100:+.1f}%")
                pc[3].metric("Peer Group Size",          f"{peer_size:,}")

                rng       = np.random.default_rng(42)
                peer_std  = max((peer_p75 - peer_p25) / 1.35, 1)
                syn_peers = rng.normal(loc=peer_median, scale=peer_std, size=500)
                syn_peers = syn_peers[(syn_peers > 0) & (syn_peers < peer_median * 4)]

                fig5 = go.Figure()
                fig5.add_histogram(x=syn_peers, nbinsx=40, marker_color="#aaaaaa", opacity=0.7,
                                    name="Peer group (modeled)",
                                    hovertemplate="$%{x:.0f}<br>Count: %{y}<extra></extra>")
                fig5.add_vrect(x0=peer_median * 0.85, x1=peer_median * 1.15,
                                fillcolor="green", opacity=0.1, line_width=0,
                                annotation_text="±15% fair zone", annotation_position="top left",
                                annotation_font_size=10)
                fig5.add_vline(x=peer_median, line_color="#333333", line_width=2, line_dash="dash",
                                annotation_text=f"Median: ${peer_median:,.0f}", annotation_position="top")
                fig5.add_vline(x=this_psqft, line_color=color, line_width=3,
                                annotation_text=f"This: ${this_psqft:,.0f}", annotation_position="bottom")
                fig5.update_layout(
                    xaxis_title="Assessed Value per Sqft ($)", yaxis_title="Count", showlegend=False,
                )
                style_fig(fig5, height=340,
                          title=f"Peer Group — {BORO_MAP.get(str(prop.get('BORO','')), '')} / {prop.get('BLDG_CLASS','')}")
                st.plotly_chart(fig5, width='stretch', config=PLOTLY_CONFIG)
                st.caption(f"25th–75th percentile: ${peer_p25:,.0f}–${peer_p75:,.0f}/sqft  |  {peer_size:,} properties in peer group")
            else:
                st.info("Not enough peer data to show comparison.")

            st.divider()
            st.subheader("Historical Classification (FY2020–FY2025)")
            hist_cols = st.columns(6)
            for i, yr in enumerate([2020, 2021, 2022, 2023, 2024, 2025]):
                over  = int(prop.get(f"overvalued_{yr}",  0) or 0)
                under = int(prop.get(f"undervalued_{yr}", 0) or 0)
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
