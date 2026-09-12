"""End-to-end pipeline for the UNSW DataSoc x Atlassian Datathon.

Run from the project folder:
    python pipeline.py --data-dir data --output-dir outputs

The target is a cancellation/refund REQUEST proxy, not confirmed churn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET = "At Risk"
CATEGORICAL_FEATURES = [
    "Product Purchased",
    "Industry",
    "Region",
    "Company Size",
    "Plan Type",
]
NUMERIC_FEATURES = [
    "Account Age Years",
    "Usage Months",
    "Total Active Days",
    "Total Sessions",
    "Total Product Actions",
    "Average Collaborators",
    "Average Integrations Used",
    "Sessions per Active Day",
    "Actions per Session",
    "Session Trend per Month",
    "Active Day Trend per Month",
    "Session Change Percent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def slope(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(clean)), clean.to_numpy(), 1)[0])


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickets = pd.read_csv(data_dir / "customer_support_tickets.csv", encoding="utf-8-sig")
    customers = pd.read_csv(data_dir / "customers.csv")
    usage = pd.read_csv(data_dir / "product_usage.csv")
    return tickets, customers, usage


def validate_sources(tickets: pd.DataFrame, customers: pd.DataFrame, usage: pd.DataFrame) -> dict:
    required = {
        "tickets": {"Ticket ID", "Customer Email", "Product Purchased", "Ticket Type"},
        "customers": {"Customer ID", "Customer Email", "Account Created Date"},
        "usage": {"Customer ID", "Product", "Month", "Sessions", "Active Days"},
    }
    frames = {"tickets": tickets, "customers": customers, "usage": usage}
    for name, columns in required.items():
        missing = columns - set(frames[name].columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    if customers["Customer ID"].duplicated().any():
        raise ValueError("Customer ID must be unique in customers.csv")
    if customers["Customer Email"].duplicated().any():
        raise ValueError("Customer Email must be unique in customers.csv")
    if tickets["Ticket ID"].duplicated().any():
        raise ValueError("Ticket ID must be unique in customer_support_tickets.csv")

    return {
        "ticket_rows": len(tickets),
        "customer_rows": len(customers),
        "usage_rows": len(usage),
        "ticket_email_match_rate": float(tickets["Customer Email"].isin(customers["Customer Email"]).mean()),
        "usage_customer_match_rate": float(usage["Customer ID"].isin(customers["Customer ID"]).mean()),
        "duplicate_ticket_rows": int(tickets.duplicated().sum()),
        "duplicate_customer_rows": int(customers.duplicated().sum()),
        "duplicate_usage_rows": int(usage.duplicated().sum()),
    }


def build_usage_features(usage: pd.DataFrame) -> pd.DataFrame:
    usage = usage.copy()
    usage["Month"] = pd.to_datetime(usage["Month"], format="%Y-%m", errors="coerce")
    usage = usage.sort_values(["Customer ID", "Product", "Month"])

    aggregate = usage.groupby(["Customer ID", "Product"], as_index=False).agg(
        **{
            "Usage Months": ("Month", "nunique"),
            "Total Active Days": ("Active Days", "sum"),
            "Total Sessions": ("Sessions", "sum"),
            "Total Product Actions": ("Product Actions", "sum"),
            "Average Collaborators": ("Collaborators", "mean"),
            "Average Integrations Used": ("Integrations Used", "mean"),
            "First Month Sessions": ("Sessions", "first"),
            "Last Month Sessions": ("Sessions", "last"),
        }
    )

    trends = (
        usage.groupby(["Customer ID", "Product"])
        .agg(
            **{
                "Session Trend per Month": ("Sessions", slope),
                "Active Day Trend per Month": ("Active Days", slope),
            }
        )
        .reset_index()
    )
    features = aggregate.merge(trends, on=["Customer ID", "Product"], validate="one_to_one")
    features["Sessions per Active Day"] = safe_divide(
        features["Total Sessions"], features["Total Active Days"]
    )
    features["Actions per Session"] = safe_divide(
        features["Total Product Actions"], features["Total Sessions"]
    )
    features["Session Change Percent"] = (
        features["Last Month Sessions"] - features["First Month Sessions"]
    ) / features["First Month Sessions"].replace(0, 1)
    return features


def build_ticket_table(
    tickets: pd.DataFrame, customers: pd.DataFrame, usage_features: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    customer_columns = [
        "Customer ID",
        "Customer Email",
        "Industry",
        "Region",
        "Company Size",
        "Plan Type",
        "Account Created Date",
    ]
    joined = tickets.merge(
        customers[customer_columns],
        on="Customer Email",
        how="left",
        validate="many_to_one",
    )
    joined = joined.merge(
        usage_features,
        left_on=["Customer ID", "Product Purchased"],
        right_on=["Customer ID", "Product"],
        how="left",
        validate="many_to_one",
    ).drop(columns="Product")

    joined["Account Created Date"] = pd.to_datetime(joined["Account Created Date"], errors="coerce")
    joined["First Response Time Parsed"] = pd.to_datetime(
        joined["First Response Time"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    joined["Time to Resolution Parsed"] = pd.to_datetime(
        joined["Time to Resolution"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    joined["Account Age Years"] = (
        joined["First Response Time Parsed"] - joined["Account Created Date"]
    ).dt.days / 365.25
    joined["Resolution Hours Raw"] = (
        joined["Time to Resolution Parsed"] - joined["First Response Time Parsed"]
    ).dt.total_seconds() / 3600
    joined["Resolution Time Valid"] = joined["Resolution Hours Raw"].ge(0)
    joined["Resolution Hours Clean"] = joined["Resolution Hours Raw"].where(
        joined["Resolution Time Valid"]
    )
    joined[TARGET] = joined["Ticket Type"].isin(
        ["Cancellation request", "Refund request"]
    ).astype(int)

    quality = {
        "joined_rows": len(joined),
        "missing_customer_id_after_join": int(joined["Customer ID"].isna().sum()),
        "missing_usage_after_join": int(joined["Total Sessions"].isna().sum()),
        "negative_resolution_durations": int(joined["Resolution Hours Raw"].lt(0).sum()),
        "satisfaction_missing_rate": float(joined["Customer Satisfaction Rating"].isna().mean()),
        "risk_proxy_rate": float(joined[TARGET].mean()),
    }
    return joined, quality


def save_eda(joined: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    segment = (
        joined.groupby(["Plan Type", "Product Purchased"], observed=True)
        .agg(
            Tickets=("Ticket ID", "count"),
            Customers=("Customer ID", "nunique"),
            Risk_Proxy_Rate=(TARGET, "mean"),
            Mean_Satisfaction=("Customer Satisfaction Rating", "mean"),
            Mean_Sessions=("Total Sessions", "mean"),
            Mean_Session_Trend=("Session Trend per Month", "mean"),
        )
        .reset_index()
    )
    segment.to_csv(output_dir / "eda_segment_summary.csv", index=False)

    type_summary = (
        joined.groupby("Ticket Type", observed=True)
        .agg(Tickets=("Ticket ID", "count"), Mean_Satisfaction=("Customer Satisfaction Rating", "mean"))
        .reset_index()
        .sort_values("Tickets", ascending=False)
    )
    type_summary.to_csv(output_dir / "eda_ticket_type_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=type_summary, x="Tickets", y="Ticket Type", ax=axes[0], color="#1868DB")
    axes[0].set_title("Ticket volume by type")
    axes[0].set_xlabel("Tickets")
    axes[0].set_ylabel("")
    plan_risk = joined.groupby("Plan Type", observed=True)[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=plan_risk.values, y=plan_risk.index, ax=axes[1], color="#6554C0")
    axes[1].set_title("Cancellation/refund request proxy by plan")
    axes[1].set_xlabel("Risk-proxy rate")
    axes[1].set_ylabel("")
    axes[1].xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    monthly = joined[[TARGET, "Customer ID", "Product Purchased"]].merge(
        # Rebuilt later from source is unnecessary; trend features already show direction.
        joined[["Ticket ID", "Session Trend per Month"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    del monthly  # Explicitly avoid a misleading pseudo-time-series chart.


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "Model": name,
        "ROC AUC": roc_auc_score(y_test, probability),
        "PR AUC": average_precision_score(y_test, probability),
        "Accuracy": accuracy_score(y_test, prediction),
        "Balanced Accuracy": balanced_accuracy_score(y_test, prediction),
        "Majority Baseline Accuracy": max(float(y_test.mean()), float(1 - y_test.mean())),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1": f1_score(y_test, prediction, zero_division=0),
        "TN": int(confusion_matrix(y_test, prediction).ravel()[0]),
        "FP": int(confusion_matrix(y_test, prediction).ravel()[1]),
        "FN": int(confusion_matrix(y_test, prediction).ravel()[2]),
        "TP": int(confusion_matrix(y_test, prediction).ravel()[3]),
    }


def train_models(joined: pd.DataFrame, output_dir: Path) -> tuple[Pipeline, pd.DataFrame]:
    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = joined[feature_columns]
    y = joined[TARGET]
    groups = joined["Customer ID"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    fitted: dict[str, Pipeline] = {}
    metrics = []
    for name, estimator in candidates.items():
        model = Pipeline([("preprocessor", make_preprocessor()), ("classifier", estimator)])
        model.fit(X_train, y_train)
        fitted[name] = model
        metrics.append(evaluate_model(name, model, X_test, y_test))

    metric_frame = pd.DataFrame(metrics).sort_values("ROC AUC", ascending=False)
    metric_frame.to_csv(output_dir / "model_metrics.csv", index=False)
    best_name = metric_frame.iloc[0]["Model"]
    best_model = fitted[str(best_name)]
    joblib.dump(best_model, output_dir / "risk_model.joblib")

    transformed_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    classifier = best_model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    else:
        importance = np.abs(classifier.coef_[0])
    importance_frame = (
        pd.DataFrame({"Feature": transformed_names, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .head(30)
    )
    importance_frame["Feature"] = (
        importance_frame["Feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    importance_frame.to_csv(output_dir / "feature_importance.csv", index=False)

    scored = joined.copy()
    scored["Risk Score"] = best_model.predict_proba(scored[feature_columns])[:, 1]
    scored["Risk Band"] = pd.cut(
        scored["Risk Score"], bins=[-0.01, 0.35, 0.60, 1.0], labels=["Low", "Medium", "High"]
    )
    value_map = {"Free": 1, "Standard": 2, "Premium": 3, "Enterprise": 4}
    scored["Value Score"] = scored["Plan Type"].map(value_map).fillna(1)
    scored["Priority Score"] = 0.7 * scored["Risk Score"] + 0.3 * (scored["Value Score"] / 4)
    scored["Recommended Action"] = np.select(
        [
            (scored["Risk Band"] == "High") & (scored["Plan Type"].isin(["Enterprise", "Premium"])),
            scored["Risk Band"] == "High",
            scored["Risk Band"] == "Medium",
        ],
        ["Specialist outreach", "Proactive support review", "Monitor and send targeted guidance"],
        default="Standard support",
    )

    public_columns = [
        "Ticket ID", "Customer ID", "Product Purchased", "Industry", "Region", "Company Size",
        "Plan Type", "Ticket Type", "Ticket Subject", "Ticket Status", "Ticket Priority",
        "Ticket Channel", "Customer Satisfaction Rating", TARGET, "Total Active Days",
        "Total Sessions", "Total Product Actions", "Average Collaborators",
        "Average Integrations Used", "Session Trend per Month", "Session Change Percent",
        "Risk Score", "Risk Band", "Value Score", "Priority Score", "Recommended Action",
    ]
    scored[public_columns].sort_values("Priority Score", ascending=False).to_csv(
        output_dir / "scored_tickets.csv", index=False
    )
    return best_model, metric_frame


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tickets, customers, usage = load_data(args.data_dir)
    source_quality = validate_sources(tickets, customers, usage)
    usage_features = build_usage_features(usage)
    joined, join_quality = build_ticket_table(tickets, customers, usage_features)

    joined.to_csv(args.output_dir / "analysis_dataset.csv", index=False)
    save_eda(joined, args.output_dir)
    _, metrics = train_models(joined, args.output_dir)

    quality = {**source_quality, **join_quality}
    (args.output_dir / "data_quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    print("Pipeline completed.")
    print(json.dumps(quality, indent=2))
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
