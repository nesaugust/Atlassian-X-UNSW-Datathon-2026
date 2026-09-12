"""Streamlit executive dashboard for the Atlassian customer-risk prototype."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


OUTPUT_DIR = Path(__file__).parent / "outputs"
ATLASSIAN_BLUE = "#0C66E4"
ATLASSIAN_NAVY = "#172B4D"
ATLASSIAN_PURPLE = "#6554C0"
PANEL = "#FFFFFF"
TEXT = "#172B4D"

st.set_page_config(page_title="Customer Support Early Warning", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #F7F8FA; color: #172B4D; }
    html, body, [class*="css"] { font-size: 16px; }
    [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * {
        color: #172B4D;
    }
    [data-testid="stSidebar"] { background: #F1F3F6; border-right: 1px solid #DFE1E6; }
    [data-testid="stHeader"] { background: rgba(247,248,250,.94); }
    [data-testid="stSidebar"] * { color: #172B4D; }
    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #DFE1E6; border-radius: 10px;
        padding: 16px 18px; box-shadow: 0 2px 8px rgba(9,30,66,.08);
    }
    [data-testid="stMetricLabel"] { color: #344563 !important; font-size: 0.95rem; font-weight: 650; }
    [data-testid="stMetricValue"] { color: #172B4D !important; font-size: 2rem; font-weight: 700; }
    .block-container { max-width: 1450px; padding-top: 2.1rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #172B4D; letter-spacing: -0.02em; }
    p, label, .stCaption, [data-testid="stCaptionContainer"] {
        color: #344563 !important; font-size: 0.95rem;
    }
    [data-testid="stSidebar"] label { color: #172B4D !important; font-weight: 600; }
    div[data-baseweb="select"] > div {
        background: #FFFFFF !important; border-color: #8590A2 !important; color: #172B4D !important;
    }
    div[data-baseweb="select"] input { color: #172B4D !important; }
    span[data-baseweb="tag"] { background: #DDEBFF !important; }
    span[data-baseweb="tag"], span[data-baseweb="tag"] * { color: #0747A6 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; border-bottom: 1px solid #DFE1E6; }
    .stTabs [data-baseweb="tab"] { color: #344563 !important; font-size: 0.95rem; font-weight: 650; }
    .stTabs [aria-selected="true"] { color: #0C66E4 !important; }
    [data-testid="stAlert"] { background: #E9F2FF; border: 1px solid #B3D4FF; color: #172B4D; }
    [data-testid="stDataFrame"] { border: 1px solid #DFE1E6; border-radius: 8px; }
    .stButton > button, .stDownloadButton > button {
        background: #0C66E4; color: #FFFFFF; border: 0; border-radius: 6px; font-weight: 600;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #0055CC; color: #FFFFFF; border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Customer Support Early Warning")
st.caption(
    "Explores whether product usage and customer context can identify proactive support opportunities "
    "before a cancellation or refund request."
)


@st.cache_data
def load_outputs():
    scored = pd.read_csv(OUTPUT_DIR / "scored_tickets.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "model_metrics.csv")
    importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv")
    return scored, metrics, importance


try:
    data, metrics, importance = load_outputs()
except FileNotFoundError:
    st.error("Run `python pipeline.py --data-dir data --output-dir outputs` first.")
    st.stop()

best_auc = metrics["ROC AUC"].max()
if best_auc < 0.60:
    baseline = (
        metrics["Majority Baseline Accuracy"].max()
        if "Majority Baseline Accuracy" in metrics.columns
        else max(data["At Risk"].mean(), 1 - data["At Risk"].mean())
    )
    st.info(
        f"MODEL STATUS · EXPLORATORY — ROC AUC {best_auc:.3f}; majority-class accuracy baseline {baseline:.1%}. "
        "Current features do not reliably identify cancellation/refund requests. Use the interface to demonstrate "
        "the decision workflow and data requirements, not automated customer decisions."
    )


def polish_chart(fig):
    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family="Arial, sans-serif", size=14, color=TEXT),
        title_font_color=ATLASSIAN_NAVY,
        coloraxis_colorbar=dict(
            tickfont=dict(color=TEXT, size=12),
            title_font=dict(color=TEXT, size=12),
        ),
        margin=dict(l=25, r=25, t=65, b=30),
        legend_title_text="",
        title_font=dict(size=18, color=ATLASSIAN_NAVY),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=TEXT),
    )
    fig.update_xaxes(
        gridcolor="#DDE2E8", zerolinecolor="#C7CED8", linecolor="#8590A2",
        tickfont=dict(color="#172B4D", size=13),
        title_font=dict(color="#172B4D", size=14),
    )
    fig.update_yaxes(
        gridcolor="#DDE2E8", zerolinecolor="#C7CED8", linecolor="#8590A2",
        tickfont=dict(color="#172B4D", size=13),
        title_font=dict(color="#172B4D", size=14),
    )
    return fig

st.sidebar.header("Filters")


def multiselect(label, column):
    values = sorted(data[column].dropna().unique().tolist())
    return st.sidebar.multiselect(label, values, default=values)


products = multiselect("Product", "Product Purchased")
plans = multiselect("Plan", "Plan Type")
regions = multiselect("Region", "Region")
priorities = multiselect("Ticket priority", "Ticket Priority")

filtered = data[
    data["Product Purchased"].isin(products)
    & data["Plan Type"].isin(plans)
    & data["Region"].isin(regions)
    & data["Ticket Priority"].isin(priorities)
].copy()

tab1, tab2, tab3 = st.tabs(["Executive overview", "Priority explorer", "Model and caveats"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets", f"{len(filtered):,}")
    c2.metric("Customers", f"{filtered['Customer ID'].nunique():,}")
    c3.metric("High review-score band", f"{(filtered['Risk Band'] == 'High').mean():.1%}")
    c4.metric("Cancel/refund requests", f"{filtered['At Risk'].mean():.1%}")

    left, right = st.columns(2)
    plan = filtered.groupby("Plan Type", as_index=False).agg(
        Tickets=("Ticket ID", "count"), Risk_Proxy_Rate=("At Risk", "mean")
    )
    fig_plan = px.bar(
            plan.sort_values("Risk_Proxy_Rate", ascending=False),
            x="Plan Type", y="Risk_Proxy_Rate", color="Tickets",
            title="Observed cancellation/refund request rate by plan",
            labels={"Risk_Proxy_Rate": "Request rate"},
            color_continuous_scale=["#DEEBFF", ATLASSIAN_BLUE],
    )
    left.plotly_chart(polish_chart(fig_plan), use_container_width=True)

    product = filtered.groupby("Product Purchased", as_index=False).agg(
        Request_Rate=("At Risk", "mean"), Customers=("Customer ID", "nunique")
    )
    fig_product = px.bar(
            product.sort_values("Request_Rate", ascending=False),
            x="Product Purchased", y="Request_Rate", color_discrete_sequence=[ATLASSIAN_PURPLE],
            title="Observed cancellation/refund request rate by product",
            labels={"Request_Rate": "Request rate"},
    )
    fig_product.update_yaxes(tickformat=".0%")
    right.plotly_chart(polish_chart(fig_product), use_container_width=True)

    action = filtered["Recommended Action"].value_counts().rename_axis("Action").reset_index(name="Tickets")
    fig_action = px.bar(
            action, x="Tickets", y="Action", orientation="h",
            title="Recommended support actions", color_discrete_sequence=[ATLASSIAN_BLUE],
    )
    st.plotly_chart(polish_chart(fig_action), use_container_width=True)

with tab2:
    st.subheader("Experimental support-review queue")
    st.caption("For workflow demonstration only; human review is required and model discrimination is weak.")
    minimum_score = st.slider("Minimum priority score", 0.0, 1.0, 0.60, 0.05)
    queue = filtered[filtered["Priority Score"] >= minimum_score].sort_values(
        "Priority Score", ascending=False
    )
    st.dataframe(
        queue[[
            "Customer ID", "Ticket ID", "Product Purchased", "Plan Type", "Company Size",
            "Risk Score", "Risk Band", "Priority Score", "Session Trend per Month",
            "Ticket Status", "Recommended Action",
        ]].style.format({"Risk Score": "{:.1%}", "Priority Score": "{:.1%}", "Session Trend per Month": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download filtered action queue",
        queue.to_csv(index=False).encode("utf-8"),
        "support_action_queue.csv",
        "text/csv",
    )

with tab3:
    st.subheader("Held-out model performance")
    st.dataframe(metrics.style.format({
        "ROC AUC": "{:.3f}", "PR AUC": "{:.3f}", "Accuracy": "{:.3f}",
        "Balanced Accuracy": "{:.3f}", "Majority Baseline Accuracy": "{:.3f}",
        "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}",
    }), use_container_width=True, hide_index=True)
    st.info(
        "The target is whether a ticket is a cancellation or refund request. "
        "It is a prioritisation proxy, not a verified churn label."
    )
    top = importance.head(15).sort_values("Importance")
    fig_importance = px.bar(
            top, x="Importance", y="Feature", orientation="h",
            title="Top model signals", color_discrete_sequence=[ATLASSIAN_PURPLE],
    )
    st.plotly_chart(polish_chart(fig_importance), use_container_width=True)
    st.markdown(
        """
        **Important limitations**

        - Satisfaction is recorded only for a subset of tickets and is not used as a pre-ticket model feature.
        - Negative resolution durations are excluded from duration analysis.
        - Product usage covers January–May 2023; the model treats it as a pre-support behavioural window.
        - Ticket outcomes, renewals, revenue and confirmed churn are unavailable.
        - Weak held-out discrimination should lead to better data collection, not exaggerated claims.
        """
    )
