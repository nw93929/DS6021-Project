from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Binary Logistic Regression", page_icon="🧮")

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cleaned"

DATA_CONFIG = {
    "Batters": {
        "file": "final_batters_df.csv",
        "features": [
            "age",
            "position",
            "AB",
            "R",
            "H",
            "2B",
            "3B",
            "HR",
            "RBI",
            "SB",
            "CS",
            "BB",
            "SO",
            "IBB",
            "HBP",
            "SH",
            "SF",
            "GIDP",
            "all_star",
            "won_mvp",
            "won_gold_glove",
            "won_silver_slugger",
        ],
        "intro": (
            "This binary logistic regression predicts whether MLB batters sign short-term "
            "(1–2 year) or long-term (3+ year) contracts. The original notebook achieved "
            "≈88% cross‑validated accuracy and AUC ≈0.898."
        ),
        "highlights": [
            "Strongest positive signals: Hits, RBI, Silver Slugger, Sacrifice Flies.",
            "Strongest negatives: Age and GIDP push players toward shorter deals.",
            "L1 (Lasso) regularization trims less-informative stats like AB, SB, and most positions.",
        ],
        "longform": """
### Model Performance
- 5-fold CV Accuracy ≈ 88.1% (Log Loss ≈ 0.283)
- Holdout Accuracy ≈ 79.1%; AUC-ROC ≈ 0.898
- Confusion: 74/95 short-term and 13/15 long-term classified correctly in the notebook run.

### Key Predictors
- **Hits (H)** and **RBI** drive longer deals; **Silver Slugger** adds a modest bump.
- **Age** is the dominant drag on long-term deals; **GIDP** also hurts odds.
- Lasso dropped 14 of 22 features; performance volume (AB, R, SB) mattered less than efficiency.

### Interpretation & Limitations
- Teams pay for youth and production efficiency; raw playing time alone is not rewarded.
- Data is 2003-era; modern Statcast-style metrics are absent, and class imbalance persists (86/14 split).
""",
    },
    "Pitchers": {
        "file": "final_pitchers_df.csv",
        "features": [
            "age",
            "position",
            "W",
            "L",
            "G",
            "GS",
            "CG",
            "SHO",
            "SV",
            "H",
            "ER",
            "HR",
            "BB",
            "SO",
            "IBB",
            "HBP",
            "BK",
            "BFP",
            "GF",
            "R",
            "SH",
            "SF",
            "GIDP",
            "ERA",
            "BAOpp",
            "InnOuts",
            "all_star",
            "won_cy_young",
            "won_mvp",
            "won_gold_glove",
        ],
        "intro": (
            "This model classifies pitcher contracts into short- vs. long-term deals. The "
            "notebook reported ≈87% CV accuracy and AUC ≈0.741, with clear emphasis on wins, "
            "durability, and age."
        ),
        "highlights": [
            "Top positives: Wins, Games pitched, Complete Games, Strikeouts, Saves.",
            "Top negatives: Age and ERA shrink long-term odds; awards largely dropped to zero.",
            "L1 regularization retained 16 of 30 features, emphasizing workload and run prevention.",
        ],
        "longform": """
### Model Performance
- 5-fold CV Accuracy ≈ 87.1% (Log Loss ≈ 0.335)
- Holdout Accuracy ≈ 66.7%; AUC-ROC ≈ 0.741
- Test confusion (notebook): 55/84 short-term and 11/15 long-term predicted correctly.

### Key Predictors
- **Wins (W)** and **Games (G)** are the strongest long-term signals; **CG** and **SO** add lift.
- **Age** and **ERA** sharply reduce long-term odds; **SHO** and **IBB** also trend negative.
- Awards (All-Star, Cy Young, MVP) were eliminated—performance metrics captured their signal.

### Interpretation & Limitations
- Durability plus run prevention outweigh accolades; age bias is strong beyond 30–32.
- Data is limited to 2003-era stats with no role split (starter vs. reliever) or injury history.
""",
    },
}


@st.cache_data
def load_dataset(file_name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / file_name)


def build_contract_binary(length: float):
    if pd.isna(length):
        return np.nan
    if length <= 2:
        return 0
    return 1


def clean_feature_name(raw: str) -> str:
    if raw.startswith("encoder__position_"):
        return f"Position: {raw.split('encoder__position_', 1)[1]}"
    return raw.split("__", 1)[-1]


def prepare_features(df: pd.DataFrame, feature_cols: list[str]):
    df = df.copy()
    df["contract_binary"] = df["contract_length"].apply(build_contract_binary)
    df = df.dropna(subset=["contract_binary", "position"])

    X = df[feature_cols]
    y = df["contract_binary"].astype(int)

    cat_cols = ["position"]
    num_cols = [col for col in feature_cols if col != "position"]

    preprocess = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(drop="first"), cat_cols),
            ("numeric", StandardScaler(), num_cols),
        ],
        sparse_threshold=0,
    )
    return X, y, preprocess


@st.cache_data(show_spinner=False)
def run_logistic_workflow(player_type: str):
    cfg = DATA_CONFIG[player_type]
    df = load_dataset(cfg["file"])
    X, y, preprocess = prepare_features(df, cfg["features"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    base_model = Pipeline(
        [("prep", preprocess), ("logit", LogisticRegression(class_weight="balanced", max_iter=2000))]
    )
    base_model.fit(X_train, y_train)
    proba_test = base_model.predict_proba(X_test)
    pred_test = base_model.predict(X_test)

    holdout_metrics = {
        "accuracy": accuracy_score(y_test, pred_test),
        "log_loss": log_loss(y_test, proba_test),
        "confusion": confusion_matrix(y_test, pred_test),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_pipe = Pipeline([("prep", preprocess), ("logit", LogisticRegression(max_iter=2000))])
    cv_results = cross_validate(
        cv_pipe, X, y, cv=cv, scoring={"acc": "accuracy", "neg_log_loss": "neg_log_loss"}
    )
    cv_metrics = {
        "accuracy": np.mean(cv_results["test_acc"]),
        "log_loss": -np.mean(cv_results["test_neg_log_loss"]),
        "fold_acc": cv_results["test_acc"],
    }

    lasso = Pipeline(
        [
            ("prep", preprocess),
            (
                "logit",
                LogisticRegressionCV(
                    penalty="l1",
                    solver="saga",
                    Cs=25,
                    cv=cv,
                    scoring="neg_log_loss",
                    max_iter=8000,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    lasso.fit(X_train, y_train)
    lasso_model = lasso.named_steps["logit"]
    feature_names = [clean_feature_name(f) for f in lasso.named_steps["prep"].get_feature_names_out()]
    coefs = lasso_model.coef_.ravel()
    coef_df = (
        pd.DataFrame({"Feature": feature_names, "Coefficient": coefs, "Abs": np.abs(coefs)})
        .sort_values("Abs", ascending=False)
        .reset_index(drop=True)
    )
    coef_df["Odds Ratio"] = np.exp(coef_df["Coefficient"])

    prob_long = lasso.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob_long)
    auc = roc_auc_score(y_test, prob_long)

    return {
        "df": df,
        "X": X,
        "y": y,
        "holdout": holdout_metrics,
        "cv": cv_metrics,
        "coef_df": coef_df,
        "roc": {"fpr": fpr, "tpr": tpr, "auc": auc},
    }


def build_coef_fig(coef_df: pd.DataFrame, title: str):
    fig = px.bar(
        coef_df.head(15),
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale=["red", "white", "green"],
        title=title,
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=650,
        xaxis_title="Coefficient (Positive = Longer Contracts)",
        yaxis_title="Feature",
    )
    return fig


def build_roc_fig(fpr, tpr, auc, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.3f})"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random (0.5)",
            line=dict(color="gray", dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        height=550,
        showlegend=True,
    )
    return fig


st.title("Binary Logistic Regression – Contract Length Classification")
st.write(
    "Predicting whether MLB free agents sign short-term (1–2 years) or long-term (3+ years) "
    "contracts using traditional performance stats, demographics, and awards."
)

player_type = st.selectbox("Choose which players to analyze:", list(DATA_CONFIG.keys()))
cfg = DATA_CONFIG[player_type]
st.header(player_type)
st.markdown(cfg["intro"])

results = run_logistic_workflow(player_type)

col1, col2, col3 = st.columns(3)
col1.metric("Rows (post-cleaning)", f"{len(results['X']):,}")
col2.metric("Class 1 share (Long-term)", f"{results['y'].mean()*100:.1f}%")
col3.metric("Feature count", f"{len(cfg['features'])}")

st.subheader("Performance Snapshots")
perf_col1, perf_col2 = st.columns(2)
perf_col1.metric("5-fold CV Accuracy", f"{results['cv']['accuracy']*100:.1f}%")
perf_col1.metric("5-fold CV Log Loss", f"{results['cv']['log_loss']:.3f}")
perf_col2.metric("Holdout Accuracy", f"{results['holdout']['accuracy']*100:.1f}%")
perf_col2.metric("Holdout Log Loss", f"{results['holdout']['log_loss']:.3f}")
perf_col2.metric("Holdout AUC (L1 model)", f"{results['roc']['auc']:.3f}")

st.caption(f"Fold accuracies: {', '.join(f'{x*100:.1f}%' for x in results['cv']['fold_acc'])}")

st.subheader("Confusion Matrix (Holdout)")
cm = results["holdout"]["confusion"]
cm_fig = px.imshow(
    cm,
    text_auto=True,
    labels={"x": "Predicted", "y": "Actual"},
    x=["Short", "Long"],
    y=["Short", "Long"],
    color_continuous_scale="Blues",
)
cm_fig.update_layout(height=400, coloraxis_showscale=False)
st.plotly_chart(cm_fig, use_container_width=True)

st.subheader("Most Important Features (L1)")
st.plotly_chart(
    build_coef_fig(results["coef_df"], f"{player_type}: Lasso Coefficients"),
    use_container_width=True,
)

st.subheader("Odds Ratios (L1 model)")
pos_table = results["coef_df"].query("Coefficient > 0").head(8)
neg_table = results["coef_df"].query("Coefficient < 0").head(8)
odds_col1, odds_col2 = st.columns(2)
odds_col1.markdown("**Top Positive Predictors**")
odds_col1.dataframe(pos_table[["Feature", "Coefficient", "Odds Ratio"]], hide_index=True)
odds_col2.markdown("**Top Negative Predictors**")
odds_col2.dataframe(neg_table[["Feature", "Coefficient", "Odds Ratio"]], hide_index=True)

st.subheader("ROC Curve")
st.plotly_chart(
    build_roc_fig(
        results["roc"]["fpr"],
        results["roc"]["tpr"],
        results["roc"]["auc"],
        f"{player_type} ROC – Long-term vs Short-term",
    ),
    use_container_width=True,
)

st.subheader("Notebook Highlights")
st.markdown("• " + "\n• ".join(cfg["highlights"]))

with st.expander("Detailed narrative and limitations"):
    st.markdown(cfg["longform"])

with st.expander("Model interpretation notes from notebook"):
    if player_type == "Batters":
        st.markdown(
            """
**Key findings (Batters)**
- Age is the dominant negative factor; older hitters see sharply lower odds of long-term deals.
- Hits and RBI are the strongest positives; situational hitting (SF) helps more than raw opportunities.
- Awards add moderate lift (Silver Slugger), while many volume stats (AB, SB) were zeroed out by L1.
"""
        )
    else:
        st.markdown(
            """
**Key findings (Pitchers)**
- Wins and games pitched dominate, signaling that durability and perceived run support matter.
- Age and ERA cut long-term odds roughly in half per unit; awards were mostly discarded.
- Complete games and strikeouts help; role-specific nuances (starter vs. reliever) are not modeled.
"""
        )
