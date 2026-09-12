# Atlassian Customer Support Early-Warning Prototype

This project joins customer support, customer profile and product usage data; performs EDA; trains leakage-aware classification models; scores support priorities; and serves an executive Streamlit dashboard.

The model target is a **cancellation/refund request proxy**. It is not confirmed churn.

## Project structure

```text
atlassian_datathon_project/
├── app.py
├── pipeline.py
├── requirements.txt
├── data/
│   ├── customer_support_tickets.csv
│   ├── customers.csv
│   └── product_usage.csv
└── outputs/                  # created by pipeline.py
```

## Set up

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the analysis and modelling pipeline

```bash
python pipeline.py --data-dir data --output-dir outputs
```

Generated outputs include:

- `analysis_dataset.csv`: ticket-level joined table
- `data_quality_report.json`: join and source checks
- `eda_segment_summary.csv`: segment-level EDA
- `eda_overview.png`: presentation-ready overview chart
- `model_metrics.csv`: held-out performance
- `feature_importance.csv`: model signals
- `risk_model.joblib`: fitted pipeline
- `scored_tickets.csv`: de-identified dashboard data and actions

## Launch the dashboard

```bash
streamlit run app.py
```

## Analytical choices

- Tickets are joined to customers by unique customer email because tickets have no Customer ID.
- Monthly usage is aggregated to `Customer ID + Product` before joining to avoid multiplying ticket rows.
- Train/test splitting is grouped by Customer ID to prevent the same customer appearing in both sets.
- Personally identifying names and emails are excluded from the dashboard export.
- Satisfaction and ticket-resolution outcomes are excluded from model inputs because they are not known before support handling.
- Negative resolution durations are marked invalid rather than silently converted.

## Datathon interpretation

Present the dashboard as a prototype that answers:

1. Which customers should support teams review first?
2. What observable signals caused the prioritisation?
3. What action should the team take?
4. What additional outcome data is needed before production deployment?

Do not overstate performance. If the held-out ROC AUC is close to 0.5, the responsible conclusion is that the supplied pre-ticket features do not reliably predict cancellation/refund requests. Recommend collecting ticket text, renewal outcomes, subscription value, prior support history and longer usage history, then validate through a controlled pilot.

For the supplied files, the best held-out ROC AUC is approximately **0.514**. Logistic regression accuracy is about **50.1%**, below the approximately **58.5%** majority-class baseline in the held-out customer groups. The balanced class weighting deliberately trades raw accuracy for detection of the positive class, but ROC AUC confirms that the available predictors contain little separative signal. Therefore the included review queue is an exploratory workflow prototype, not a deployable automated decision system.
