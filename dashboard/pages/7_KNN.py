from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="KNN Regression", page_icon="🧭")

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cleaned"

DATA_CONFIG = {
    "Batters": {
        "file": "final_batters_df.csv",
        "drop_cols": [
            "row_id",
            "playerID",
            "year",
            "free_agent_salary",
            "ZR",
            "position",
            "WP",
            "all_star",
            "contract_length",
            "won_cy_young",
            "won_mvp",
            "won_gold_glove",
            "won_silver_slugger",
        ],
        "notebook_summary": (
            "Test R² ≈ 0.68 and RMSE ≈ $2.8M. The model predicts well on lower salaries but "
            "overestimates some of the highest contracts. Best K = 3 points to a model that is "
            "fairly local while still generalizing across players."
        ),
    },
    "Pitchers": {
        "file": "final_pitchers_df.csv",
        "drop_cols": [
            "row_id",
            "playerID",
            "year",
            "free_agent_salary",
            "ZR",
            "position",
            "H",
            "BFP",
            "R",
            "GIDP",
            "ERA",
            "PB",
            "WP.1",
            "won_cy_young",
            "won_mvp",
            "won_gold_glove",
            "won_silver_slugger",
            "all_star",
        ],
        "notebook_summary": (
            "Test R² ≈ 0.59 and RMSE ≈ $2.4M. Predictions are reasonable across the salary range "
            "but can overestimate higher-end deals. Best K = 3 suggests the model benefits from a "
            "fairly local neighborhood focus."
        ),
    },
}


@st.cache_data
def load_dataset(file_name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / file_name)


def prepare_features(df: pd.DataFrame, drop_cols: list[str]):
    df_clean = df.dropna(subset=["free_agent_salary"]).copy()
    X = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
    X = X.select_dtypes(include=[np.number])
    y = df_clean["free_agent_salary"]
    return X, y


def run_knn_model(X: pd.DataFrame, y: pd.Series):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Match the notebook flow: impute and scale once on the training split, then grid-search KNN.
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    knn = KNeighborsRegressor()
    param_grid = {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]}
    grid = GridSearchCV(knn, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid.fit(X_train_scaled, y_train)

    best_knn = grid.best_estimator_
    y_pred = best_knn.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return {
        "grid": grid,
        "y_test": y_test,
        "y_pred": y_pred,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def build_actual_vs_pred_plot(y_test, y_pred, title: str):
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "Actual Free Agent Salary", "y": "Predicted Salary"},
        title=title,
    )
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="firebrick", dash="dash"),
    )
    fig.update_layout(height=600)
    return fig


st.title("K-Nearest Neighbors Salary Prediction")
st.markdown(
    """
KNN regression predicts MLB free agent salaries using a median-impute → standardize → KNN pipeline.
This page mirrors the **KNN - Lasso Selection** notebooks for batters and pitchers, including the
hyperparameter search and actual-vs-predicted visualization.
"""
)

st.sidebar.header("KNN Regression")
choice = st.selectbox("Choose which players to analyze:", list(DATA_CONFIG.keys()))
config = DATA_CONFIG[choice]

st.header("📋 Notebook Notes")
st.markdown(config["notebook_summary"])

with st.spinner("Loading data and preparing features..."):
    df = load_dataset(config["file"])
    X, y = prepare_features(df, config["drop_cols"])

st.write(f"Rows after cleaning: **{len(X)}**, numeric features used: **{X.shape[1]}**")

with st.spinner("Running grid search and fitting KNN..."):
    results = run_knn_model(X, y)

best_params = results["grid"].best_params_
col1, col2, col3 = st.columns(3)
col1.metric("Best k", best_params["n_neighbors"])
col2.metric("Weights", best_params["weights"])
col3.metric("CV Folds", "5")

metric_cols = st.columns(3)
metric_cols[0].metric("Test R²", f"{results['r2']:.3f}")
metric_cols[1].metric("Test MSE", f"{results['mse']:.2f}")
metric_cols[2].metric("Test RMSE", f"{results['rmse']:.0f}")

st.plotly_chart(
    build_actual_vs_pred_plot(
        results["y_test"],
        results["y_pred"],
        f"Actual vs Predicted Salaries — {choice} (KNN Regression)",
    ),
    use_container_width=True,
)

st.header("ℹ️ How this matches the notebook")
st.markdown(
    """
- **Feature selection**: Dropped the same ID/target/award columns and used numeric features only.
- **Preprocessing**: Median imputation and standardization identical to the notebook.
- **Hyperparameter search**: Grid search over k = [3,5,7,9,11] and weights ∈ {uniform, distance}.
- **Evaluation**: 80/20 train-test split with R², MSE/RMSE, and the diagonal scatter plot from the notebook.
"""
)

st.caption(
    "R² and RMSE may vary slightly from the static notebook outputs due to randomized train/test split, "
    "but the workflow and hyperparameters match the recorded analysis."
)
