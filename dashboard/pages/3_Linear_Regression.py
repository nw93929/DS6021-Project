from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.api as sm

def compute_pvalues_transformed(model, X, y):
    """
    Compute OLS p-values using the ALREADY-FITTED pipeline preprocessor.
    """
    # use fitted preprocessor from the model
    preprocessor = model.named_steps["prep"]

    # IMPORTANT: only transform — do NOT fit again
    X_transformed = preprocessor.transform(X)

    # Get clean feature names
    feature_names = preprocessor.get_feature_names_out()

    # Convert to DataFrame
    X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    # Add intercept
    X_df = sm.add_constant(X_df)

    # Fit OLS
    ols_model = sm.OLS(y, X_df).fit()

    return ols_model.pvalues, ols_model.params, feature_names




st.set_page_config(page_title="Linear Regression", page_icon="📈")

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
            "free_agent_salary_log",
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
    },
    "Pitchers": {
        "file": "final_pitchers_df.csv",
        "drop_cols": [
            "row_id",
            "playerID",
            "year",
            "free_agent_salary",
            "free_agent_salary_log",
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
    },
}


@st.cache_data
def load_dataset(file_name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / file_name)


def clean_feature_names(feature_names: np.ndarray) -> list[str]:
    return [name.split("__", 1)[-1] for name in feature_names]


def prepare_features(df: pd.DataFrame, drop_cols: list[str]):
    df = df.dropna(subset=["free_agent_salary"]).copy()
    df["free_agent_salary_log"] = np.log1p(df["free_agent_salary"])

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["free_agent_salary_log"]

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = [("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols)]
    if cat_cols:
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    return X, y, preprocessor


def run_linear_regression(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = Pipeline(
        [
            ("prep", preprocessor),
            ("scale", StandardScaler()),
            ("linreg", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Convert back to salary dollars for an interpretable error metric
    y_test_dollars = np.expm1(y_test)
    y_pred_dollars = np.expm1(y_pred)
    dollar_rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))

    feature_names = clean_feature_names(model.named_steps["prep"].get_feature_names_out())
    coef_df = (
        pd.DataFrame({"feature": feature_names, "coefficient": model.named_steps["linreg"].coef_})
        .sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
        .head(12)
    )

    return {
        "model": model,
        "y_test": y_test_dollars,
        "y_pred": y_pred_dollars,
        "metrics": {"r2": r2, "mse": mse, "rmse": rmse, "dollar_rmse": dollar_rmse},
        "coef_df": coef_df,
    }


def run_lasso_selection(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    lasso = Pipeline(
        [
            ("prep", preprocessor),
            ("scale", StandardScaler()),
            ("lasso", LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=5000)),
        ]
    )
    lasso.fit(X_train, y_train)

    best_alpha = lasso.named_steps["lasso"].alpha_
    alphas = lasso.named_steps["lasso"].alphas_
    mse_path = lasso.named_steps["lasso"].mse_path_

    feature_names = clean_feature_names(lasso.named_steps["prep"].get_feature_names_out())
    raw_coefs = lasso.named_steps["lasso"].coef_
    coef_df = (
        pd.DataFrame({"feature": feature_names, "coefficient": raw_coefs})
        .query("coefficient != 0")
        .sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
    )

    return {
        "alpha": best_alpha,
        "alphas": alphas,
        "mse_path": mse_path,
        "coef_df": coef_df,
        "n_nonzero": len(coef_df),
        "n_features": len(raw_coefs),
    }


def build_actual_vs_pred_plot(y_true, y_pred, title: str):
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={"x": "Actual Salary ($)", "y": "Predicted Salary ($)"},
        title=title,
    )
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
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


def build_alpha_mse_plot(alphas: np.ndarray, mse_path: np.ndarray, best_alpha: float):
    mean_mse = mse_path.mean(axis=1)
    fig = px.line(
        x=alphas,
        y=mean_mse,
        markers=True,
        title="Lasso alpha selection (5-fold CV)",
        labels={"x": "alpha (log scale)", "y": "Mean CV MSE"},
    )
    fig.update_layout(xaxis_type="log", height=400)
    fig.add_vline(x=best_alpha, line_dash="dash", line_color="firebrick", annotation_text=f"alpha={best_alpha:.4f}")
    return fig


def build_lasso_coef_bar(coef_df: pd.DataFrame, title: str):
    fig = px.bar(
        coef_df.head(15),
        x="feature",
        y="coefficient",
        title=title,
    )
    fig.update_layout(xaxis_tickangle=-45, height=450)
    return fig


st.title("Linear Regression Salary Prediction")
st.markdown(
    """
In order to predict MLB free agent salaries, we implement a linear regression model following feature selection via LassoCV. The target variable, `free_agent_salary`, 
is log-transformed to better approximate normality. The modeling pipeline consists of median-imputation and one-hot encoding of categorical variables, standardization, LassoCV (5-fold) for variable selection, and finally fitting/testing an ordinary least squares regression (80/20 split).
"""
)

st.sidebar.header("Linear Regression")
st.sidebar.write("Switch between batters and pitchers to rerun the workflow.")

choice = st.selectbox("Choose which players to analyze:", list(DATA_CONFIG.keys()))
config = DATA_CONFIG[choice]

with st.spinner("Loading data and building features..."):
    df = load_dataset(config["file"])
    X, y, preprocessor = prepare_features(df, config["drop_cols"])

st.write(f"Rows after dropping missing salaries: **{len(X)}**, features used: **{X.shape[1]}**")

st.header("🔍 Lasso variable selection")
with st.spinner("Running Lasso variable selection..."):
    lasso_results = run_lasso_selection(X, y, preprocessor)

lasso_cols = st.columns(3)
lasso_cols[0].metric("Best alpha", f"{lasso_results['alpha']:.4f}")
lasso_cols[1].metric("Non-zero coefficients", f"{lasso_results['n_nonzero']} / {lasso_results['n_features']}")
lasso_cols[2].metric("CV folds", "5")

st.plotly_chart(
    build_alpha_mse_plot(lasso_results["alphas"], lasso_results["mse_path"], lasso_results["alpha"]),
    use_container_width=True,
)
st.plotly_chart(
    build_lasso_coef_bar(
        lasso_results["coef_df"],
        f"Lasso-selected coefficients — {choice}",
    ),
    use_container_width=True,
)

st.header("📈 Linear regression model")
with st.spinner("Training linear regression..."):
    results = run_linear_regression(X, y, preprocessor)

metric_cols = st.columns(4)
metric_cols[0].metric("Test R²", f"{results['metrics']['r2']:.3f}")
metric_cols[1].metric("Log RMSE", f"{results['metrics']['rmse']:.3f}")
metric_cols[2].metric("Log MSE", f"{results['metrics']['mse']:.3f}")
metric_cols[3].metric("Dollar RMSE", f"${results['metrics']['dollar_rmse']:,.0f}")

st.plotly_chart(
    build_actual_vs_pred_plot(
        results["y_test"],
        results["y_pred"],
        f"Actual vs Predicted Salaries — {choice} (Linear Regression)",
    ),
    use_container_width=True,
)


import statsmodels.api as sm

# Compute p-values & coefficients using statsmodels
pvalues, params, used_features = compute_pvalues_transformed(results["model"], X, y)


pval_df = pd.DataFrame({
    "feature": ["const"] + list(used_features),
    "coefficient": params.values,
    "p_value": pvalues.values
})

pval_df = pd.DataFrame({
    "feature": ["const"] + list(X.columns),
    "coefficient": params.values,
    "p_value": pvalues.values
})

st.header("🔍 Interactive Feature Exploration")

selected_feature = st.selectbox(
    "Choose a feature to analyze:",
    X.columns
)

selected_row = pval_df[pval_df["feature"] == selected_feature].iloc[0]

st.subheader("Feature Statistics")
c1, c2 = st.columns(2)
c1.metric("Coefficient", f"{selected_row['coefficient']:.4f}")
c2.metric("P-value", f"{selected_row['p_value']:.4e}")

# Scatterplot
fig = px.scatter(
    x=X[selected_feature],
    y=y,
    trendline="ols",
    labels={"x": selected_feature, "y": "free_agent_salary_log"},
    title=f"{selected_feature} vs free_agent_salary_log"
)

st.plotly_chart(fig, use_container_width=True)
