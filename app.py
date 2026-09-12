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

ATLASSIAN_MARK = (
    '<svg width="27" height="27" viewBox="0 0 32 32" role="img" aria-hidden="true">'
    '<path fill="#1868DB" d="M15.7 4.2c-.5 0-.9.3-1.1.8L4.8 24.7c-.3.7.2 1.5 1 1.5h5.1c.4 0 .8-.2 1-.6l5.9-12.1c.2-.5.2-1 0-1.5L16.8 5c-.1-.5-.6-.8-1.1-.8Z"/>'
    '<path fill="#1868DB" d="M18.5 9.4c-.4 0-.8.3-1 .7l-2.3 5.2c-.2.4-.2.8 0 1.2l4.5 9.1c.2.4.6.6 1 .6h5.4c.8 0 1.3-.8 1-1.5L19.6 10c-.2-.4-.6-.6-1.1-.6Z"/>'
    '</svg>'
)


st.set_page_config(
    page_title="Customer Support Early Warning",
    page_icon="🔷",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --background: #F7F8F9;
        --surface: #FFFFFF;
        --border: #DFE1E6;
        --text: #172B4D;
        --text-subtle: #44546F;
        --blue: #0C66E4;
        --blue-hover: #0055CC;
        --blue-subtle: #E9F2FF;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Arial, sans-serif;
        font-size: 16px;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(247, 248, 249, 0.94);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    /* Main dashboard header */

    .brand-lockup {
        background: linear-gradient(120deg, #FFFFFF 0%, #F4F9FF 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 22px 26px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(9, 30, 66, 0.08);
    }

    .brand-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    }

    .brand-wordmark {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #1868DB;
    }

    .brand-wordmark span {
        font-size: 0.82rem;
        font-weight: 750;
        letter-spacing: 0.10em;
    }

    .datathon-label {
        color: #626F86;
        font-size: 0.76rem;
        font-weight: 650;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .brand-lockup h1 {
        color: var(--text);
        font-size: 2.25rem;
        line-height: 1.15;
        letter-spacing: -0.035em;
        margin: 18px 0 7px;
    }

    .brand-lockup p {
        color: var(--text-subtle);
        font-size: 0.98rem;
        margin: 0;
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.02em;
    }

    p, label, .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-subtle) !important;
        font-size: 0.95rem;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background-color: #F7F8F9;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #1868DB;
        padding: 2px 0 14px;
        margin-bottom: 14px;
        border-bottom: 1px solid var(--border);
    }

    .sidebar-brand strong {
        font-size: 0.80rem;
        letter-spacing: 0.10em;
    }

    .sidebar-kicker {
        color: #626F86;
        font-size: 0.70rem;
        font-weight: 650;
        letter-spacing: 0.055em;
        text-transform: uppercase;
        margin: -4px 0 16px;
    }

    .filter-intro {
        color: #44546F;
        font-size: 0.78rem;
        line-height: 1.45;
        margin-bottom: 12px;
    }

    [data-testid="stSidebar"] h2 {
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0;
        padding: 0 0 0.35rem;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: var(--text) !important;
        font-weight: 600;
        font-size: 0.86rem;
    }

    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stMultiSelect {
        margin-bottom: 14px;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(9, 30, 66, 0.06);
        margin-bottom: 10px;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: var(--text);
        font-size: 0.88rem;
        font-weight: 650;
    }

    /* Multiselect fields */

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text) !important;
        min-height: 42px;
        box-shadow: none !important;
    }

    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 1px var(--blue) !important;
    }

    div[data-baseweb="select"] input {
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] [data-baseweb="tag"],
    span[data-baseweb="tag"] {
        background-color: var(--blue-subtle) !important;
        color: var(--blue) !important;
        border: 1px solid #B6D8FF !important;
        border-radius: 5px !important;
    }

    [data-testid="stSidebar"] [data-baseweb="tag"] span,
    span[data-baseweb="tag"] span {
        color: #0747A6 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="tag"] svg,
    span[data-baseweb="tag"] svg {
        fill: var(--blue) !important;
    }

    /* KPI cards */

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(9, 30, 66, 0.10);
    }

    [data-testid="stMetricLabel"] {
        color: #44546F !important;
        font-size: 0.86rem;
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.75rem;
        font-weight: 720;
    }

    /* Tabs */

    .stTabs [data-baseweb="tab-list"] {
        gap: 18px;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-subtle) !important;
        font-size: 0.90rem;
        font-weight: 650;
        padding-left: 0;
        padding-right: 0;
    }

    .stTabs [aria-selected="true"] {
        color: var(--blue) !important;
    }

    /* Alerts, tables and charts */

    [data-testid="stAlert"] {
        background: var(--blue-subtle);
        border: 1px solid #B6D8FF;
        border-left: 4px solid var(--blue);
        border-radius: 6px;
        color: var(--text);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    .stPlotlyChart {
        background: var(--surface);
        border: 1px solid #EBECF0;
        border-radius: 8px;
        overflow: hidden;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: var(--blue);
        color: #FFFFFF;
        border: 0;
        border-radius: 6px;
        font-weight: 650;
        min-height: 40px;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: var(--blue-hover);
        color: #FFFFFF;
        border: 0;
    }

    @media (max-width: 900px) {
        .brand-row {
            align-items: flex-start;
            flex-direction: column;
        }

        .brand-lockup h1 {
            font-size: 1.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Main dashboard header and embedded logo
st.markdown(
    '<div class="brand-lockup">'
    '<div class="brand-row">'
    f'<div class="brand-wordmark" aria-label="Atlassian">{ATLASSIAN_MARK}<span>ATLASSIAN</span></div>'
    '<div class="datathon-label">UNSW DataSoc × Atlassian Datathon 2026</div>'
    '</div>'
    '<h1>Customer Support Early Warning</h1>'
    '<p>Explore whether product usage and customer context can identify proactive support opportunities before a cancellation or refund request.</p>'
    '</div>',
    unsafe_allow_html=True,
)


@st.cache_data
def load_outputs():
    scored = pd.read_csv(OUTPUT_DIR / "scored_tickets.csv")

    # Support both the current Review Tier output and older Risk Band output.
    if "Review Tier" not in scored.columns and "Risk Band" in scored.columns:
        scored["Review Tier"] = scored["Risk Band"].map(
            {
                "High": "Tier 1 · Top 10%",
                "Medium": "Tier 2 · Next 20%",
                "Low": "Tier 3 · Remaining 70%",
            }
        )

    if "Risk Band" not in scored.columns and "Review Tier" in scored.columns:
        tier_number = (
            scored["Review Tier"]
            .astype(str)
            .str.extract(r"Tier\s+(\d)")[0]
        )

        scored["Risk Band"] = tier_number.map(
            {
                "1": "High",
                "2": "Medium",
                "3": "Low",
            }
        )

    # Fallback if neither column exists.
    if "Review Tier" not in scored.columns:
        percentile = scored["Risk Score"].rank(
            method="first",
            ascending=False,
            pct=True,
        )

        scored["Review Tier"] = pd.cut(
            percentile,
            bins=[0, 0.10, 0.30, 1.00],
            labels=[
                "Tier 1 · Top 10%",
                "Tier 2 · Next 20%",
                "Tier 3 · Remaining 70%",
            ],
            include_lowest=True,
        ).astype(str)

    if "Risk Band" not in scored.columns:
        scored["Risk Band"] = scored["Review Tier"].map(
            {
                "Tier 1 · Top 10%": "High",
                "Tier 2 · Next 20%": "Medium",
                "Tier 3 · Remaining 70%": "Low",
            }
        )

    if "Value Score" not in scored.columns:
        value_map = {
            "Free": 1,
            "Standard": 2,
            "Premium": 3,
            "Enterprise": 4,
        }
        scored["Value Score"] = (
            scored["Plan Type"]
            .map(value_map)
            .fillna(1)
        )

    if "Observed Signals" not in scored.columns:
        scored["Observed Signals"] = "Usage context available for review"

    metrics = pd.read_csv(OUTPUT_DIR / "model_metrics.csv")
    importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv")

    return scored, metrics, importance


try:
    data, metrics, importance = load_outputs()
except FileNotFoundError:
    st.error(
        "Required output files were not found. Run "
        "`python pipeline.py --data-dir data --output-dir outputs` first."
    )
    st.stop()


best_auc = metrics["ROC AUC"].max()

if best_auc < 0.60:
    baseline = (
        metrics["Majority Baseline Accuracy"].max()
        if "Majority Baseline Accuracy" in metrics.columns
        else max(data["At Risk"].mean(), 1 - data["At Risk"].mean())
    )

    st.info(
        f"MODEL STATUS · EXPLORATORY — ROC AUC {best_auc:.3f}; "
        f"majority-class accuracy baseline {baseline:.1%}. "
        "Current features do not reliably identify cancellation/refund requests. "
        "Use the interface to demonstrate the decision workflow and data "
        "requirements, not automated customer decisions."
    )


def polish_chart(fig):
    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={
            "family": "Arial, sans-serif",
            "size": 14,
            "color": TEXT,
        },
        coloraxis_colorbar={
            "tickfont": {"color": TEXT, "size": 12},
            "title_font": {"color": TEXT, "size": 12},
        },
        margin={"l": 25, "r": 25, "t": 65, "b": 30},
        legend_title_text="",
        title_font={"size": 18, "color": ATLASSIAN_NAVY},
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "font_color": TEXT,
        },
    )

    fig.update_xaxes(
        gridcolor="#DDE2E8",
        zerolinecolor="#C7CED8",
        linecolor="#8590A2",
        tickfont={"color": ATLASSIAN_NAVY, "size": 13},
        title_font={"color": ATLASSIAN_NAVY, "size": 14},
    )

    fig.update_yaxes(
        gridcolor="#DDE2E8",
        zerolinecolor="#C7CED8",
        linecolor="#8590A2",
        tickfont={"color": ATLASSIAN_NAVY, "size": 13},
        title_font={"color": ATLASSIAN_NAVY, "size": 14},
    )

    return fig


# Sidebar logo and filters
st.sidebar.markdown(
    f'<div class="sidebar-brand" aria-label="Atlassian">{ATLASSIAN_MARK}<strong>ATLASSIAN</strong></div>'
    '<div class="sidebar-kicker">Customer Support Intelligence</div>',
    unsafe_allow_html=True,
)

st.sidebar.header("Dashboard filters")

st.sidebar.markdown(
    '<div class="filter-intro">Leave a field blank to include all values.</div>',
    unsafe_allow_html=True,
)


def multiselect(container, label, column):
    values = sorted(data[column].dropna().unique().tolist())

    return container.multiselect(
        label,
        values,
        default=[],
    )


account_filters = st.sidebar.expander(
    "Customer and product",
    expanded=True,
)

products = multiselect(
    account_filters,
    "Product",
    "Product Purchased",
)

plans = multiselect(
    account_filters,
    "Plan",
    "Plan Type",
)

regions = multiselect(
    account_filters,
    "Region",
    "Region",
)

industries = multiselect(
    account_filters,
    "Industry",
    "Industry",
)

company_sizes = multiselect(
    account_filters,
    "Company size",
    "Company Size",
)


ticket_filters = st.sidebar.expander(
    "Support ticket",
    expanded=True,
)

priorities = multiselect(
    ticket_filters,
    "Ticket priority",
    "Ticket Priority",
)


active_filter_count = sum(
    bool(selection)
    for selection in [
        products,
        plans,
        regions,
        industries,
        company_sizes,
        priorities,
    ]
)

st.sidebar.caption(
    f"{active_filter_count} active "
    f"filter{'s' if active_filter_count != 1 else ''} · "
    "blank fields include all"
)


def selected_or_all(column, selection):
    if not selection:
        return pd.Series(True, index=data.index)

    return data[column].isin(selection)


filtered = data[
    selected_or_all("Product Purchased", products)
    & selected_or_all("Plan Type", plans)
    & selected_or_all("Region", regions)
    & selected_or_all("Industry", industries)
    & selected_or_all("Company Size", company_sizes)
    & selected_or_all("Ticket Priority", priorities)
].copy()


if filtered.empty:
    st.warning(
        "No tickets match the selected filters. "
        "Adjust or clear one or more filters."
    )
    st.stop()


tab1, tab2, tab3 = st.tabs(
    [
        "Executive overview",
        "Priority explorer",
        "Model and caveats",
    ]
)


with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Tickets",
        f"{len(filtered):,}",
    )

    c2.metric(
        "Customers",
        f"{filtered['Customer ID'].nunique():,}",
    )

    c3.metric(
        "Tier 1 review queue",
        f"{(filtered['Review Tier'] == 'Tier 1 · Top 10%').mean():.1%}",
    )

    c4.metric(
        "Cancel/refund requests",
        f"{filtered['At Risk'].mean():.1%}",
    )

    c5.metric(
        "Satisfaction coverage",
        f"{filtered['Customer Satisfaction Rating'].notna().mean():.1%}",
    )

    left, right = st.columns(2)

    plan = (
        filtered.groupby("Plan Type", as_index=False)
        .agg(
            Tickets=("Ticket ID", "count"),
            Request_Rate=("At Risk", "mean"),
        )
    )

    fig_plan = px.bar(
        plan.sort_values("Request_Rate", ascending=False),
        x="Plan Type",
        y="Request_Rate",
        color="Tickets",
        title="Observed cancellation/refund request rate by plan",
        labels={
            "Request_Rate": "Request rate",
            "Plan Type": "Plan",
        },
        color_continuous_scale=[
            "#DEEBFF",
            ATLASSIAN_BLUE,
        ],
    )

    fig_plan.update_yaxes(tickformat=".0%")

    left.plotly_chart(
        polish_chart(fig_plan),
        use_container_width=True,
    )

    product = (
        filtered.groupby("Product Purchased", as_index=False)
        .agg(
            Request_Rate=("At Risk", "mean"),
            Customers=("Customer ID", "nunique"),
        )
    )

    fig_product = px.bar(
        product.sort_values("Request_Rate", ascending=False),
        x="Product Purchased",
        y="Request_Rate",
        title="Observed cancellation/refund request rate by product",
        labels={
            "Request_Rate": "Request rate",
            "Product Purchased": "Product",
        },
        color_discrete_sequence=[ATLASSIAN_PURPLE],
    )

    fig_product.update_yaxes(tickformat=".0%")

    right.plotly_chart(
        polish_chart(fig_product),
        use_container_width=True,
    )

    action = (
        filtered["Recommended Action"]
        .value_counts()
        .rename_axis("Action")
        .reset_index(name="Tickets")
    )

    fig_action = px.bar(
        action.sort_values("Tickets"),
        x="Tickets",
        y="Action",
        orientation="h",
        title="Recommended support actions",
        color_discrete_sequence=[ATLASSIAN_BLUE],
    )

    st.plotly_chart(
        polish_chart(fig_action),
        use_container_width=True,
    )


with tab2:
    st.subheader("Experimental support-review queue")

    st.caption(
        "Review tiers allocate workload by relative score: Tier 1 is the "
        "top 10% and Tier 2 is the next 20%. They are not calibrated risk "
        "levels, and human review is required."
    )

    min_priority = float(filtered["Priority Score"].min())
    max_priority = float(filtered["Priority Score"].max())
    default_priority = float(
        filtered["Priority Score"].quantile(0.70)
    )

    if max_priority > min_priority:
        minimum_score = st.slider(
            "Minimum priority score",
            min_value=min_priority,
            max_value=max_priority,
            value=min(
                max(default_priority, min_priority),
                max_priority,
            ),
            step=max(
                (max_priority - min_priority) / 100,
                0.001,
            ),
            format="%.3f",
        )
    else:
        minimum_score = min_priority

        st.caption(
            f"All filtered tickets have the same priority score: "
            f"{minimum_score:.3f}"
        )

    queue = (
        filtered[
            filtered["Priority Score"] >= minimum_score
        ]
        .sort_values(
            "Priority Score",
            ascending=False,
        )
    )

    st.caption(
        "Priority Score = 70% exploratory Risk Score + "
        "30% normalised Value Score. Value Score is a transparent plan "
        "proxy from 1 (Free) to 4 (Enterprise). Observed Signals are "
        "descriptive review prompts, not causal explanations."
    )

    queue_columns = [
        "Customer ID",
        "Ticket ID",
        "Product Purchased",
        "Plan Type",
        "Company Size",
        "Risk Score",
        "Review Tier",
        "Value Score",
        "Priority Score",
        "Session Trend per Month",
        "Observed Signals",
        "Ticket Status",
        "Recommended Action",
    ]

    # Only display columns available in the current output file.
    queue_columns = [
        column
        for column in queue_columns
        if column in queue.columns
    ]

    queue_display = queue[queue_columns].style.format(
        {
            "Risk Score": "{:.1%}",
            "Value Score": "{:.0f}",
            "Priority Score": "{:.1%}",
            "Session Trend per Month": "{:.2f}",
        }
    )

    st.dataframe(
        queue_display,
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

    metric_formats = {
        "ROC AUC": "{:.3f}",
        "PR AUC": "{:.3f}",
        "Accuracy": "{:.3f}",
        "Balanced Accuracy": "{:.3f}",
        "Majority Baseline Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1": "{:.3f}",
    }

    metric_formats = {
        column: formatting
        for column, formatting in metric_formats.items()
        if column in metrics.columns
    }

    st.dataframe(
        metrics.style.format(metric_formats),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The target is whether a ticket is a cancellation or refund request. "
        "It is a prioritisation proxy, not a verified churn label."
    )

    top = (
        importance.head(15)
        .sort_values("Importance")
    )

    fig_importance = px.bar(
        top,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top model signals",
        color_discrete_sequence=[ATLASSIAN_PURPLE],
    )

    st.plotly_chart(
        polish_chart(fig_importance),
        use_container_width=True,
    )

    st.markdown(
        """
        **Important limitations**

        - Satisfaction is recorded only for a subset of tickets and is not
          used as a pre-ticket model feature.
        - Negative resolution durations are excluded from duration analysis.
        - Product usage covers January–May 2023; the model treats it as a
          pre-support behavioural window.
        - Ticket outcomes, renewals, revenue and confirmed churn are unavailable.
        - Review tiers are quantile-based workload allocations, not calibrated
          risk categories.
        - Observed Signals are deterministic usage summaries rather than causal
          or SHAP explanations.
        - Weak held-out discrimination should lead to better data collection,
          not exaggerated claims.
        """
    )
