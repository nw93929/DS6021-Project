from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor


st.set_page_config(page_title="Exploratory Data Analysis (Notebook)", page_icon="📈")

sns.set_theme(style="darkgrid")

DATA_CONFIG = {
    "Batters": {
        "file": "final_batters_df.csv",
        "note": "Distribution and correlations for hitters in free agency.",
    },
    "Pitchers": {
        "file": "final_pitchers_df.csv",
        "note": "Distribution and correlations for pitchers in free agency.",
    },
}

DROP_COLS = ["playerID", "year", "free_agent_salary", "free_agent_salary_log", "row_id", "ZR"]


@st.cache_data
def load_dataset(file_name: str) -> pd.DataFrame:
    # File resides under dashboard/pages; climb to project root to read data/cleaned
    data_dir = Path(__file__).resolve().parents[2] / "data" / "cleaned"
    return pd.read_csv(data_dir / file_name)


def add_log_salary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["free_agent_salary_log"] = np.log1p(df["free_agent_salary"])
    return df


def plot_salary_histogram(df: pd.DataFrame, column: str, title: str):
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=column, bins=30, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_heatmap_with_target(df: pd.DataFrame, target: str):
    fig_height = min(12, len(df.columns) * 0.25 + 1)  # limit height so bar stays compact
    fig, ax = plt.subplots(figsize=(3.8, fig_height))
    corr = df.corr(numeric_only=True)[[target]].sort_values(by=target, ascending=False)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(f"Correlations with {target}")
    fig.tight_layout()
    return fig


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


st.title("Exploratory Data Analysis")

st.header("📋 What you'll see")
st.markdown(
    """
This page recreates the key exploratory visuals from the EDA notebook for both batters and pitchers:

- Raw free agent salary distribution
- Log-transformed salary distribution
- Correlations of each feature with log salary
- Variance Inflation Factors (VIF) to assess multicollinearity
"""
)

st.header("🧭 How to use")
st.markdown(
    """
Choose a group to load the cleaned dataset, apply the same log transformation, and render the exact
seaborn/matplotlib visuals used in the notebook.
"""
)

st.sidebar.header("EDA")
st.sidebar.write("Switch between batters and pitchers to recreate the notebook plots.")

choice = st.selectbox("Choose which players to analyze:", list(DATA_CONFIG.keys()))
config = DATA_CONFIG[choice]

df_raw = load_dataset(config["file"])
df = add_log_salary(df_raw)

st.subheader(f"{choice} Dataset Overview")
st.write(config["note"])
st.markdown(
    """
Context from the notebook:
- Raw salary is sharply right-skewed with many small contracts and only a few multi-million deals.
- Applying `log1p` produces a more normal-looking target for modeling and stabilizes variance.
- Correlations highlight which performance stats move with free-agent pay (positive for strong production, negative for weaker outcomes/aging).
- Extremely high VIF values signal heavy multicollinearity, motivating PCA/Lasso in later modeling.
"""
)

cols = st.columns(3)
cols[0].metric("Rows", f"{len(df):,}")
cols[1].metric("Columns", f"{df.shape[1]}")
cols[2].metric("Log Column", "free_agent_salary_log")

st.markdown("---")

st.subheader("Salary Distributions")
st.pyplot(plot_salary_histogram(df, "free_agent_salary", "free_agent_salary"), use_container_width=True)
st.pyplot(plot_salary_histogram(df, "free_agent_salary_log", "free_agent_salary_log"), use_container_width=True)
st.caption(
    "Notebook takeaway: raw salaries pile up at low values; the log transform normalizes the response so linear models fit better."
)

st.subheader("Correlations with Log Salary")
st.pyplot(plot_heatmap_with_target(df, "free_agent_salary_log"), use_container_width=True)
st.caption(
    "Notebook takeaway: strongest positive relationships come from productive stats (e.g., runs/RBI for batters, strikeouts/innings for pitchers); "
    "negative links show costly outcomes (e.g., ERA/OppBA) or aging effects."
)

st.subheader("Variance Inflation Factors (VIF)")
vif_table = compute_vif(df)
st.dataframe(vif_table.style.format({"VIF": "{:.2f}"}), use_container_width=True)
st.caption(
    "Notebook takeaway: very high VIFs confirm heavy multicollinearity across performance stats, reinforcing the need for PCA or regularization."
)
