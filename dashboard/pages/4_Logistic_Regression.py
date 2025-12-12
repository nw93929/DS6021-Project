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


def derive_form_features(coef_df: pd.DataFrame) -> list[str]:
    """Map coefficient feature names to user-facing inputs (collapse one-hot positions)."""
    form_features: list[str] = []
    for feature in coef_df.loc[coef_df["Coefficient"] != 0, "Feature"]:
        key = "position" if feature.startswith("Position:") else feature
        if key not in form_features:
            form_features.append(key)
    return form_features


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


@st.cache_resource(show_spinner=False)
def train_final_lasso_model(player_type: str):
    cfg = DATA_CONFIG[player_type]
    df = load_dataset(cfg["file"])
    X, y, preprocess = prepare_features(df, cfg["features"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = Pipeline(
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
    model.fit(X, y)
    feature_names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["logit"].coef_.ravel()

    active_features = set()
    for raw_name, coef in zip(feature_names, coefs):
        if coef == 0:
            continue
        if raw_name.startswith("encoder__position_"):
            active_features.add("position")
        else:
            active_features.add(raw_name.split("__", 1)[-1])

    return model, sorted(active_features)


def build_feature_defaults(df: pd.DataFrame, feature_cols: list[str]):
    defaults: dict[str, int | str] = {}
    bounds: dict[str, tuple[int, int]] = {}

    for col in feature_cols:
        series = df[col].dropna()
        if col == "position":
            defaults[col] = series.mode().iat[0] if not series.empty else ""
            continue

        if series.empty:
            defaults[col] = 0
            bounds[col] = (0, 100)
            continue

        defaults[col] = int(round(series.median()))
        bounds[col] = (int(series.min()), int(series.max()))

    return defaults, bounds


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
    active = coef_df.loc[coef_df["Coefficient"] != 0]
    if active.empty:
        active = coef_df
    fig = px.bar(
        active,
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


# ==================== STREAMLIT UI ====================
st.title("Binary Logistic Regression – Contract Length Classification")
st.write(
    "Predicting whether MLB free agents sign short-term (1–2 years) or long-term (3+ years) "
    "contracts using traditional performance stats, demographics, and awards."
)

player_type = st.selectbox("Choose which players to analyze:", list(DATA_CONFIG.keys()))
cfg = DATA_CONFIG[player_type]

results = run_logistic_workflow(player_type)

# ==================== 1. DESCRIPTION ABOUT THE MODEL ====================
st.header(f"📊 {player_type} Model Overview")
st.markdown(cfg["intro"])

data_col1, data_col2, data_col3 = st.columns(3)
data_col1.metric("Dataset Size", f"{len(results['X']):,} rows")
data_col2.metric("Long-term Contracts", f"{results['y'].mean()*100:.1f}%")
data_col3.metric("Total Features", f"{len(cfg['features'])}")

st.divider()

# ==================== 4. TRAIN TEST OUTPUT ====================
st.header("📈 Train-Test Split Results")
st.markdown("**Holdout Test Set (20%)** – Base Logistic Regression Model")

test_col1, test_col2, test_col3 = st.columns(3)
test_col1.metric("Accuracy", f"{results['holdout']['accuracy']*100:.2f}%", 
                 help="Percentage of correctly classified contracts")
test_col2.metric("Log Loss", f"{results['holdout']['log_loss']:.4f}",
                 help="Lower is better; measures probability calibration")
test_col3.metric("AUC-ROC (L1 model)", f"{results['roc']['auc']:.4f}",
                 help="Area under ROC curve; measures discrimination ability")

st.divider()

# ==================== 5. REGULARIZATION OUTPUT ====================
st.header("🎚️ Regularization (L1/Lasso) Output")
st.markdown("""
**L1 (Lasso) Regularization** performs automatic feature selection by shrinking less important coefficients to zero.
This helps identify the most predictive features and reduces overfitting.
""")

non_zero_coefs = (results["coef_df"]["Coefficient"] != 0).sum()
zero_coefs = len(results["coef_df"]) - non_zero_coefs

reg_col1, reg_col2, reg_col3 = st.columns(3)
reg_col1.metric("Features Retained", f"{non_zero_coefs}")
reg_col2.metric("Features Eliminated", f"{zero_coefs}")
reg_col3.metric("Retention Rate", f"{non_zero_coefs/len(results['coef_df'])*100:.1f}%")

st.divider()

# ==================== 6. CROSS-VALIDATION OUTPUT ====================
st.header("🔄 Cross-Validation Results")
st.markdown("**5-Fold Stratified Cross-Validation** – Base Logistic Regression Model")

cv_col1, cv_col2 = st.columns(2)
cv_col1.metric("Mean CV Accuracy", f"{results['cv']['accuracy']*100:.2f}%",
               help="Average accuracy across 5 folds")
cv_col2.metric("Mean CV Log Loss", f"{results['cv']['log_loss']:.4f}",
               help="Average log loss across 5 folds")

st.markdown("**Individual Fold Performance:**")
fold_df = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(5)],
    "Accuracy": [f"{acc*100:.2f}%" for acc in results['cv']['fold_acc']]
})
st.dataframe(fold_df, hide_index=True, use_container_width=True)

st.divider()

# ==================== 7. PREDICTIONS - CONFUSION MATRIX ====================
st.header("🎯 Predictions – Confusion Matrix")
st.markdown("**Holdout Test Set Performance** (20% of data, stratified split)")

cm = results["holdout"]["confusion"]
cm_fig = px.imshow(
    cm,
    text_auto=True,
    labels={"x": "Predicted Label", "y": "Actual Label"},
    x=["Short-term (1-2 yr)", "Long-term (3+ yr)"],
    y=["Short-term (1-2 yr)", "Long-term (3+ yr)"],
    color_continuous_scale="Blues",
)
cm_fig.update_layout(height=450, coloraxis_showscale=False)
st.plotly_chart(cm_fig, use_container_width=True)

st.caption(f"✓ Correctly classified short-term: {cm[0,0]} | ✗ Misclassified as long-term: {cm[0,1]}")
st.caption(f"✗ Misclassified as short-term: {cm[1,0]} | ✓ Correctly classified long-term: {cm[1,1]}")

st.divider()

# ==================== 8. COEFFICIENT GRAPH ====================
st.header("📊 Coefficient Visualization")
st.markdown("Top 15 features by absolute coefficient magnitude (L1 Regularized Model)")

st.plotly_chart(
    build_coef_fig(results["coef_df"], f"{player_type}: Lasso Coefficients"),
    use_container_width=True,
)

st.divider()

# ==================== 9. ROC CURVE ====================
st.header("📉 ROC Curve")
st.markdown("""
The ROC curve shows the trade-off between True Positive Rate (sensitivity) and False Positive Rate 
across different classification thresholds. AUC (Area Under Curve) summarizes overall discrimination ability.
""")

st.plotly_chart(
    build_roc_fig(
        results["roc"]["fpr"],
        results["roc"]["tpr"],
        results["roc"]["auc"],
        f"{player_type} ROC – Long-term vs Short-term Contracts",
    ),
    use_container_width=True,
)

auc_interpretation = "Excellent" if results['roc']['auc'] > 0.9 else "Good" if results['roc']['auc'] > 0.8 else "Fair" if results['roc']['auc'] > 0.7 else "Poor"
st.caption(f"**AUC = {results['roc']['auc']:.3f}** – Model discrimination is **{auc_interpretation}**")

# ==================== 2B. INTERACTIVE PREDICTION TOOL ====================
st.divider()
st.header("🛠️ Try the Final Logistic Model")
st.markdown(
    "Estimate the probability of a **long-term contract (3+ years)** using the final L1 logistic model. "
    "Inputs are restricted to whole numbers; decimals will be rounded."
)

model, model_features = train_final_lasso_model(player_type)
coef_inputs = results["coef_df"]
form_features = derive_form_features(coef_inputs) or model_features
defaults_all, bounds_all = build_feature_defaults(results["df"], cfg["features"])
defaults_final, bounds_final = build_feature_defaults(results["df"], form_features)
positions = sorted(results["df"]["position"].dropna().unique().tolist())
pos_default = defaults_all.get("position", positions[0] if positions else "")

with st.form(f"{player_type}_prediction_form"):
    if "position" in form_features:
        position_choice = st.selectbox(
            "Position",
            positions,
            index=positions.index(pos_default) if pos_default in positions else 0,
        )
    else:
        position_choice = pos_default

    numeric_features = [feat for feat in form_features if feat != "position"]
    cols = st.columns(3)
    numeric_inputs: dict[str, int] = {}

    for idx, feature in enumerate(numeric_features):
        min_val, max_val = bounds_final.get(feature, (0, 500))
        default_val = defaults_final.get(feature, min_val)
        numeric_inputs[feature] = cols[idx % 3].number_input(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=1,
            format="%d",
        )

    submitted = st.form_submit_button("Predict Contract Type")

if submitted:
    sample = {**defaults_all}
    sample["position"] = position_choice
    sample.update({feat: int(val) for feat, val in numeric_inputs.items()})
    pred_df = pd.DataFrame([sample], columns=cfg["features"])

    prob_long = float(model.predict_proba(pred_df)[0, 1])
    prob_short = 1 - prob_long
    predicted_label = "Long-term (3+ yr)" if prob_long >= 0.5 else "Short-term (1–2 yr)"

    prob_col1, prob_col2 = st.columns(2)
    prob_col1.metric("P(Long-term, 3+ yr)", f"{prob_long*100:.1f}%")
    prob_col2.metric("P(Short-term, 1–2 yr)", f"{prob_short*100:.1f}%")
    st.progress(prob_long)
    st.caption(f"Predicted class: **{predicted_label}**")

st.caption("📌 **Data Source:** MLB contract and performance data (2003 era) | **Model:** Scikit-learn Logistic Regression with L1 regularization")
