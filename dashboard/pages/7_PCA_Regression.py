from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PCA Regression", page_icon="📊")


DATA_CONFIG = {
    "Batters": {
        "file": "final_batters_df.csv",
        "n_components": 5,
    },
    "Pitchers": {
        "file": "final_pitchers_df.csv",
        "n_components": 4,
    },
}
CAT_COLS = [
    "won_mvp",
    "won_gold_glove",
    "won_cy_young",
    "position",
    "won_silver_slugger",
    "all_star",
]
DROP_COLS = ["row_id", "playerID", "year", "free_agent_salary", "ZR"]


@st.cache_data
def load_dataset(file_name: str) -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parents[2] / "data" / "cleaned"
    df = pd.read_csv(data_dir / file_name)
    return df


def prepare_features(df: pd.DataFrame):
    # Keep the original dataframe intact (like the notebook does with batters/pitchers)
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    # Create features and targets (notebook line 33-35)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df["free_agent_salary"]
    
    # Only select numerical features for PCA (notebook line 38-44)
    X = X.select_dtypes(include=[np.number])
    
    # now recombine with y and drop na values
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=["free_agent_salary"])
    y = data["free_agent_salary"]
    
    return X, y, df


def compute_pca_outputs(X: pd.DataFrame):
    pca_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA())])
    pca_pipe.fit(X)

    pca_model = pca_pipe.named_steps["pca"]
    explained_var = pca_model.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var)
    n_components = len(explained_var)

    ev_df = pd.DataFrame(
        {
            "PC": np.arange(1, n_components + 1),
            "ExplainedVariance": explained_var,
            "CumulativeVariance": cum_explained_var,
        }
    )

    scores = pca_pipe.transform(X)
    pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    scores_df = pd.DataFrame(scores, columns=pc_cols)

    loadings = pca_model.components_.T
    loading_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    loading_df = pd.DataFrame(loadings, index=X.columns, columns=loading_cols)
    return ev_df, scores_df, loading_df


def build_correlation_heatmap(df: pd.DataFrame, title: str):
    corr_matrix = df.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=title,
    )
    fig.update_layout(height=700)
    return fig


def build_scree_plot(ev_df: pd.DataFrame):
    fig = px.line(
        ev_df,
        x="PC",
        y="ExplainedVariance",
        markers=True,
        title="Scree Plot: Proportion of Variance Explained",
    )
    fig.update_layout(yaxis_title="Explained Variance Ratio")
    return fig


def build_biplot(scores_df: pd.DataFrame, loading_df: pd.DataFrame, ids, label: str, arrow_scale: float = 16.0, arrow_line_width: int = 3):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scores_df["PC1"],
            y=scores_df["PC2"],
            mode="markers",
            name=label,
            hoverinfo="text",
            hovertext=ids,
            marker=dict(opacity=0.7),
        )
    )
    for var_name, row in loading_df.iterrows():
        x_arrow = row["PC1"] * arrow_scale
        y_arrow = row["PC2"] * arrow_scale
        fig.add_trace(
            go.Scatter(
                x=[0, x_arrow],
                y=[0, y_arrow],
                mode="lines+markers+text",
                text=[None, var_name],
                textposition="top center",
                showlegend=False,
                line=dict(width=arrow_line_width),
                marker=dict(size=8),
                textfont=dict(size=12, color="black"),
            )
        )
    fig.update_layout(
        title="PCA Biplot: PC1 vs PC2 with Variable Directions",
        xaxis_title="PC1",
        yaxis_title="PC2",
        xaxis=dict(zeroline=True),
        yaxis=dict(zeroline=True),
        width=800,
        height=700,
    )
    return fig


def run_pcr(X: pd.DataFrame, y: pd.Series, n_components: int):
    pcr_pipe = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=n_components)), ("linreg", LinearRegression())]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pcr_pipe.fit(X_train, y_train)

    train_r2 = pcr_pipe.score(X_train, y_train)
    test_r2 = pcr_pipe.score(X_test, y_test)
    y_train_pred = pcr_pipe.predict(X_train)
    y_test_pred = pcr_pipe.predict(X_test)

    train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
    }


st.title("PCA Regression")

# Model Summary
st.header("📋 Model Summary")
st.markdown("""
**Principal Component Regression** uses PCA to perform dimensionality reduction before applying linear regression 
to predict baseball player free agent salaries. This approach addresses multicollinearity among 
performance statistics and reduces model complexity.
""")

# Results/Performance
st.header("📊 Plot Interpretations")
st.markdown("""
**Interpreting the visualizations:**
- **Correlation Matrix**: Shows relationships between original features (pre-PCA)
- **Scree Plot**: Displays variance explained by each PC (helps choose n_components)
- **Biplot**: Shows PC1 vs PC2 scores and variable loadings (reveals patterns in data)
""")

st.markdown("---")

st.markdown(
    """
Select a group below to load the cleaned dataset, run the PCA pipeline, and view the same
plots used in the original analysis.
"""
)

st.sidebar.header("PCA Regression")
st.sidebar.write("Switch between batters and pitchers to rerun the PCA workflow.")

choice = st.selectbox("Choose which players to analyze:", list(DATA_CONFIG.keys()))
config = DATA_CONFIG[choice]

# Load the dataset (like batters/pitchers in notebook)
df = load_dataset(config["file"])
X, y, original_df = prepare_features(df)

st.subheader(f"{choice} Dataset Overview")
st.write(f"Rows after cleaning: {len(X)}, features used: {X.shape[1]}")

with st.spinner("Running PCA and PCR..."):
    ev_df, scores_df, loading_df = compute_pca_outputs(X)
    results = run_pcr(X, y, n_components=config["n_components"])

metric_cols = st.columns(4)
metric_cols[0].metric("Train R²", f"{results['train_r2']:.3f}")
metric_cols[1].metric("Test R²", f"{results['test_r2']:.3f}")
metric_cols[2].metric("Train RMSE", f"{results['train_rmse']:.1f}")
metric_cols[3].metric("Test RMSE", f"{results['test_rmse']:.1f}")

st.plotly_chart(build_correlation_heatmap(X, "Correlation Matrix of Original Variables"), use_container_width=True)
st.plotly_chart(build_scree_plot(ev_df), use_container_width=True)
st.plotly_chart(
    build_biplot(
        scores_df,
        loading_df,
        original_df["playerID"],
        label=choice,
    ),
    use_container_width=True,
)

st.header("🔢 Principal Component Loadings")
st.markdown(
    "Feature weights for each principal component. Large positive/negative values indicate strong influence on that component."
)
loading_cols_to_show = [c for c in loading_df.columns if c.startswith("PC")][: config["n_components"]]
if loading_cols_to_show:
    st.dataframe(
        loading_df[loading_cols_to_show].style.background_gradient(
            cmap="RdYlGn", axis=1, subset=loading_cols_to_show, vmin=-1.0, vmax=1.0
        ),
        use_container_width=True,
    )
