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

# ==================== 2. MODEL PIPELINE ====================
st.header("🔧 Model Pipeline")
st.markdown(f"""
**Pipeline Components:**
1. **Preprocessing**
   - Categorical Encoding: One-Hot Encoding for position (drop first category)
   - Numeric Scaling: StandardScaler for all numeric features
2. **Base Model**
   - Algorithm: Logistic Regression
   - Class Weights: Balanced (to handle class imbalance)
   - Max Iterations: 2000
3. **Regularization Model**
   - Algorithm: Logistic Regression with L1 (Lasso) penalty
   - Solver: SAGA
   - Hyperparameter Tuning: 25 values of C (regularization strength)
   - Cross-Validation: 5-fold Stratified K-Fold
   - Optimization Metric: Negative Log Loss
""")

st.markdown(f"**Input Features ({len(cfg['features'])}):** {', '.join(cfg['features'][:10])}{'...' if len(cfg['features']) > 10 else ''}")

st.divider()

# ==================== 3. PREDICTIONS - CONFUSION MATRIX ====================
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

# ==================== 5. CROSS-VALIDATION OUTPUT ====================
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

# ==================== 6. REGULARIZATION OUTPUT ====================
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

# ==================== 7. TOP FEATURES COEFFICIENTS AND ODDS ====================
st.header("⭐ Top Feature Coefficients and Odds Ratios")

pos_table = results["coef_df"].query("Coefficient > 0").head(8)
neg_table = results["coef_df"].query("Coefficient < 0").head(8)

odds_col1, odds_col2 = st.columns(2)

with odds_col1:
    st.markdown("### ✅ Top Positive Predictors")
    st.markdown("*Features that increase the odds of long-term contracts*")
    st.dataframe(
        pos_table[["Feature", "Coefficient", "Odds Ratio"]].style.format({
            "Coefficient": "{:.4f}",
            "Odds Ratio": "{:.4f}"
        }),
        hide_index=True,
        use_container_width=True
    )

with odds_col2:
    st.markdown("### ❌ Top Negative Predictors")
    st.markdown("*Features that decrease the odds of long-term contracts*")
    st.dataframe(
        neg_table[["Feature", "Coefficient", "Odds Ratio"]].style.format({
            "Coefficient": "{:.4f}",
            "Odds Ratio": "{:.4f}"
        }),
        hide_index=True,
        use_container_width=True
    )

st.caption("💡 **Interpretation:** An odds ratio > 1 increases long-term contract probability; < 1 decreases it.")

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

st.divider()

# ==================== 10. FINDINGS ====================
st.header("🔍 Key Findings")
for highlight in cfg["highlights"]:
    st.markdown(f"- {highlight}")

st.divider()

# ==================== 11. INTERPRETATION ====================
st.header("💡 Model Interpretation & Limitations")

if player_type == "Batters":
    st.markdown("""
    ### Key Insights for Batters
    
    **What Drives Long-Term Contracts:**
    - **Hits (H)** and **RBI** are the strongest positive signals – teams reward offensive production
    - **Silver Slugger Award** provides a moderate boost, validating elite hitting ability
    - **Sacrifice Flies (SF)** positively correlate, suggesting clutch/situational hitting is valued
    
    **What Limits Long-Term Contracts:**
    - **Age** is the dominant negative factor – older hitters face dramatically lower odds
    - **GIDP (Grounding Into Double Plays)** hurts odds, signaling slower/less athletic players
    - Many volume stats (At-Bats, Stolen Bases) were eliminated by Lasso, showing teams care more about efficiency than raw playing time
    
    ### Model Performance Summary
    - Cross-validation accuracy: **~88%** (Log Loss: 0.283)
    - Holdout test accuracy: **~79%** (AUC: 0.898)
    - Strong class separation, though minority class (long-term) harder to predict
    
    ### Limitations
    - **Temporal bias:** Data from 2003 era; modern Statcast metrics (exit velocity, launch angle) not included
    - **Class imbalance:** ~86% short-term vs 14% long-term contracts
    - **Missing context:** No injury history, market conditions, or team-specific factors
    - **Selection bias:** Only includes players who received contracts; excludes those who retired or went unsigned
    """)
else:
    st.markdown("""
    ### Key Insights for Pitchers
    
    **What Drives Long-Term Contracts:**
    - **Wins (W)** are the strongest positive predictor – teams still value traditional success metrics
    - **Games Pitched (G)** and **Complete Games (CG)** signal durability and workhorse mentality
    - **Strikeouts (SO)** and **Saves (SV)** provide significant boosts – dominance and defined role matter
    
    **What Limits Long-Term Contracts:**
    - **Age** is the dominant negative – pitchers age poorly, risk increases dramatically after 30
    - **ERA (Earned Run Average)** sharply reduces odds – run prevention is paramount
    - **Shutouts (SHO)** and **Intentional Walks (IBB)** surprisingly negative (possibly multicollinearity)
    - **Awards** (All-Star, Cy Young, MVP) were largely eliminated – performance stats capture their signal
    
    ### Model Performance Summary
    - Cross-validation accuracy: **~87%** (Log Loss: 0.335)
    - Holdout test accuracy: **~67%** (AUC: 0.741)
    - Lower AUC than batters suggests harder discrimination task for pitchers
    
    ### Limitations
    - **Role ambiguity:** No distinction between starters and relievers (very different value propositions)
    - **Injury blind spot:** No Tommy John surgery flags or injury history
    - **Outdated metrics:** 2003-era stats; missing modern analytics (spin rate, whiff rate, hard contact %)
    - **Class imbalance:** Similar 85/15 split favoring short-term deals
    - **Market dynamics:** No team budget, positional needs, or competitive window considerations
    """)

st.divider()

st.caption("📌 **Data Source:** MLB contract and performance data (2003 era) | **Model:** Scikit-learn Logistic Regression with L1 regularization")