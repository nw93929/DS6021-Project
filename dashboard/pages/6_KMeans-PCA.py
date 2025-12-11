from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="K-Means Segmentation", page_icon="🎯")

RANDOM_STATE = 6021
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cleaned"

DATA_CONFIG = {
    "Batters": {
        "file": "final_batters_df.csv",
        "n_components": 5,
        "summary": (
            "Rate-based batting metrics (AVG, OBP, SLG, BB/SO rates, BABIP, SB efficiency) "
            "replace raw counting stats so players are compared on equal opportunity. "
            "Missingness was minimal, elbow suggested k=4 but silhouette favored k=2. "
            "PC1 captures power/on-base production, PC2 tracks age, PC3 contrasts AVG vs SLG. "
            "Power stats (SLG/ISO) drove the clearest cluster separation; salary differences are "
            "visible but contract-length gaps were negligible."
        ),
        "segment_label": "Batter Segment",
    },
    "Pitchers": {
        "file": "final_pitchers_df.csv",
        "n_components": 5,
        "summary": (
            "Pitcher segmentation uses rate stats (K%, BB%, HR%, starter share, WHIP) after "
            "dropping awards and counting totals. Missingness was minimal, elbow leaned toward "
            "k=4 while silhouette pointed to k=2. PC1 represents overall run-prevention/whiff "
            "profile, PC2 varies with workload/age, and PC3 captures contrasting run-prevention "
            "patterns. PC1–PC2 offered the cleanest split; salary/contract comparisons are shown "
            "for each pitcher cluster."
        ),
        "segment_label": "Pitcher Segment",
    },
}


@st.cache_data
def load_dataset(file_name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / file_name)


def safe_divide(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_batter_features(df: pd.DataFrame):
    X = df.select_dtypes(include=[np.number]).copy()

    drop_cols = [
        "year",
        "free_agent_salary",
        "PB",
        "WP",
        "won_cy_young",
        "won_mvp",
        "won_gold_glove",
        "won_silver_slugger",
        "all_star",
        "PO",
        "InnOuts",
        "A",
        "E",
        "ZR",
    ]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    # Rate-based features to mirror notebook
    X["PA"] = X["AB"] + X["BB"] + X["HBP"] + X["SF"] + X["SH"]
    X["SLG"] = safe_divide(
        (X["H"] - X["2B"] - X["3B"] - X["HR"]) + (2 * X["2B"]) + (3 * X["3B"]) + (4 * X["HR"]),
        X["AB"],
    )
    X["BB_rate"] = safe_divide(X["BB"], X["PA"])
    X["SO_rate"] = safe_divide(X["SO"], X["PA"])
    X["BABIP"] = safe_divide(X["H"] - X["HR"], X["AB"] - X["SO"] - X["HR"] + X["SF"])
    X["SB_EFF"] = safe_divide(X["SB"], X["SB"] + X["CS"])

    count_cols = [
        "AB",
        "PA",
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
        "DP",
    ]
    X.drop(columns=[c for c in count_cols if c in X.columns], inplace=True)

    missing_snapshot = X.copy()
    X = X.dropna()
    Xorig = X.copy()
    X_model = X.drop(columns=[c for c in ["avg_salary_year", "contract_length"] if c in X.columns])
    return X_model, Xorig, missing_snapshot


def prepare_pitcher_features(df: pd.DataFrame):
    X = df.select_dtypes(include=[np.number]).copy()

    drop_cols = [
        "year",
        "free_agent_salary",
        "won_cy_young",
        "won_mvp",
        "won_gold_glove",
        "won_silver_slugger",
        "all_star",
        "PO",
        "A",
        "E",
        "DP",
        "PB",
        "ZR",
        "WP",
        "WP.1",
        "BK",
    ]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    X["IP"] = safe_divide(X["InnOuts"], 3)
    X["K_rate"] = safe_divide(X["SO"], X["BFP"])
    X["BB_rate"] = safe_divide(X["BB"], X["BFP"])
    X["HR_rate"] = safe_divide(X["HR"], X["BFP"])
    X["GS_rate"] = safe_divide(X["GS"], X["G"])
    X["WHIP"] = safe_divide(X["H"] + X["BB"], X["IP"])

    count_cols = [
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
        "BFP",
        "GF",
        "R",
        "SH",
        "SF",
        "GIDP",
        "InnOuts",
        "IP",
    ]
    X.drop(columns=[c for c in count_cols if c in X.columns], inplace=True)

    missing_snapshot = X.copy()
    X = X.dropna()
    Xorig = X.copy()
    X_model = X.drop(columns=[c for c in ["avg_salary_year", "contract_length"] if c in X.columns])
    return X_model, Xorig, missing_snapshot


def build_missing_heatmap(df: pd.DataFrame):
    fig = px.imshow(
        df.isnull(),
        aspect="auto",
        color_continuous_scale="Greys",
        title="Missing Values (pre-dropna)",
    )
    fig.update_layout(height=400, xaxis_title="Feature", yaxis_title="Row")
    return fig


def build_elbow_plot(k_values, wcss):
    fig = px.line(
        x=k_values,
        y=wcss,
        markers=True,
        labels={"x": "Number of Clusters (k)", "y": "WCSS / Inertia"},
        title="Elbow Plot (StandardScaler → PCA → K-Means)",
    )
    fig.update_layout(height=450)
    return fig


def build_silhouette_plot(k_values, sil_scores):
    fig = px.line(
        x=k_values,
        y=sil_scores,
        markers=True,
        labels={"x": "Number of Clusters (k)", "y": "Silhouette Score"},
        title="Silhouette Scores by k",
    )
    fig.update_layout(height=450)
    return fig


def build_cumulative_variance_plot(pca: PCA):
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    fig = px.line(
        x=np.arange(1, len(cum) + 1),
        y=cum,
        markers=True,
        labels={"x": "Number of Components", "y": "Cumulative Explained Variance"},
        title="Cumulative Explained Variance by Component",
    )
    fig.update_layout(yaxis=dict(range=[0, 1]))
    return fig


def build_biplot(scores_df: pd.DataFrame, loading_df: pd.DataFrame, clusters, segment_label: str):
    arrow_scale = 6
    fig = go.Figure()

    for cluster_id in sorted(clusters.unique()):
        cluster_mask = clusters == cluster_id
        fig.add_trace(
            go.Scatter(
                x=scores_df.loc[cluster_mask, "PC1"],
                y=scores_df.loc[cluster_mask, "PC2"],
                mode="markers",
                marker=dict(size=8, opacity=0.7),
                name=f"Cluster {cluster_id}",
            )
        )

    for var_name, row in loading_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[0, row["PC1"] * arrow_scale],
                y=[0, row["PC2"] * arrow_scale],
                mode="lines+markers+text",
                text=[None, var_name],
                textposition="top center",
                line=dict(color="crimson", width=2),
                marker=dict(size=5, color="crimson"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"PCA Biplot (PC1 vs PC2) with Loadings — {segment_label}",
        xaxis_title="PC1",
        yaxis_title="PC2",
        height=650,
    )
    return fig


def build_pair_scatter(scores_df: pd.DataFrame, clusters, segment_label: str):
    dims = [c for c in scores_df.columns if c.startswith("PC")][:3]
    plot_df = scores_df[dims].copy()
    plot_df["cluster"] = clusters.astype(str)
    sns.set(style="ticks")
    pair_grid = sns.pairplot(
        plot_df,
        hue="cluster",
        diag_kind="kde",
        palette="Set2",
        plot_kws={"alpha": 0.7, "s": 40},
    )
    pair_grid.fig.suptitle(f"Pairwise PCA Scores with KDE Diagonals — {segment_label}", y=1.02)
    return pair_grid.fig


def build_feature_boxplots(X_with_clusters: pd.DataFrame, segment_label: str, standardize: bool = False):
    features_df = X_with_clusters.copy()
    if standardize:
        for col in features_df.columns:
            if col in ("cluster",):
                continue
            col_std = features_df[col].std(ddof=0)
            if col_std == 0 or np.isnan(col_std):
                continue
            features_df[col] = (features_df[col] - features_df[col].mean()) / col_std

    plot_df = features_df.melt(
        id_vars="cluster",
        value_vars=X_with_clusters.columns.difference(["cluster", "age"]),
        var_name="Feature",
        value_name="Value",
    )
    fig = px.box(
        plot_df,
        x="Feature",
        y="Value",
        color="cluster",
        title=f"Feature Distributions by Cluster — {segment_label}",
    )
    fig.update_layout(xaxis_tickangle=-45, height=600)
    return fig


def build_salary_contract_plots(df_with_clusters: pd.DataFrame, segment_label: str):
    salary_fig = px.box(
        df_with_clusters,
        x="cluster",
        y="avg_salary_year",
        color="cluster",
        title=f"Average Yearly Salary by Cluster — {segment_label}",
        labels={"avg_salary_year": "Avg Salary per Year ($)", "cluster": segment_label},
    )
    contract_fig = px.box(
        df_with_clusters,
        x="cluster",
        y="contract_length",
        color="cluster",
        title=f"Contract Length by Cluster — {segment_label}",
        labels={"contract_length": "Years", "cluster": segment_label},
    )
    return salary_fig, contract_fig


def run_t_tests(df_with_clusters: pd.DataFrame):
    results = []
    if {"avg_salary_year", "contract_length", "cluster"}.issubset(df_with_clusters.columns):
        salary0 = df_with_clusters[df_with_clusters["cluster"] == "0"]["avg_salary_year"]
        salary1 = df_with_clusters[df_with_clusters["cluster"] == "1"]["avg_salary_year"]
        t_stat, p_val = stats.ttest_ind(salary0, salary1, equal_var=False)
        results.append(("Average Salary", t_stat, p_val))

        c0 = df_with_clusters[df_with_clusters["cluster"] == "0"]["contract_length"]
        c1 = df_with_clusters[df_with_clusters["cluster"] == "1"]["contract_length"]
        t_stat_c, p_val_c = stats.ttest_ind(c0, c1, equal_var=False)
        results.append(("Contract Length", t_stat_c, p_val_c))
    return results


st.title("K-Means Segmentation")
st.markdown(
    """
Unsupervised clustering on MLB player performance using a StandardScaler → PCA → K-Means pipeline.
The workflow mirrors the project notebooks for both batters and pitchers, including elbow/silhouette
diagnostics, PCA inspection, and compensation comparisons across clusters.
"""
)

choice = st.selectbox("Choose which group to analyze:", list(DATA_CONFIG.keys()))
config = DATA_CONFIG[choice]

st.header("📋 Notebook Highlights")
st.markdown(config["summary"])

with st.spinner("Loading data and preparing features..."):
    df = load_dataset(config["file"])
    if choice == "Batters":
        X_model, Xorig, missing_snapshot = prepare_batter_features(df)
    else:
        X_model, Xorig, missing_snapshot = prepare_pitcher_features(df)

st.write(f"Rows after cleaning: **{len(X_model)}**, features used: **{X_model.shape[1]}**")

# Missingness view
st.plotly_chart(build_missing_heatmap(missing_snapshot), use_container_width=True)

# Elbow and silhouette diagnostics
with st.spinner("Running clustering diagnostics..."):
    base_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=config["n_components"], random_state=RANDOM_STATE)),
            ("kmeans", KMeans(init="k-means++", n_init="auto", random_state=RANDOM_STATE)),
        ]
    )

    k_values = list(range(1, 11))
    wcss = []
    for k in k_values:
        base_pipe.set_params(kmeans__n_clusters=k)
        base_pipe.fit(X_model)
        wcss.append(base_pipe["kmeans"].inertia_)

    sil_scores = []
    sil_k = list(range(2, 11))
    for k in sil_k:
        base_pipe.set_params(kmeans__n_clusters=k)
        base_pipe.fit(X_model)
        sil_scores.append(silhouette_score(X_model, base_pipe["kmeans"].labels_))

st.plotly_chart(build_elbow_plot(k_values, wcss), use_container_width=True)
st.plotly_chart(build_silhouette_plot(sil_k, sil_scores), use_container_width=True)

# Fit final model with k=2 as in notebook
with st.spinner("Fitting PCA + K-Means model..."):
    final_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=config["n_components"], random_state=RANDOM_STATE)),
            ("kmeans", KMeans(n_clusters=2, init="k-means++", n_init="auto", random_state=RANDOM_STATE)),
        ]
    )
    final_pipe.fit(X_model)
    labels = final_pipe["kmeans"].labels_.astype(str)
    pca_step: PCA = final_pipe["pca"]
    scores = final_pipe[:-1].transform(X_model)
    pc_cols = [f"PC{i}" for i in range(1, scores.shape[1] + 1)]
    scores_df = pd.DataFrame(scores, columns=pc_cols)
    scores_df["cluster"] = labels

    loading_df = pd.DataFrame(
        pca_step.components_.T, index=X_model.columns, columns=pc_cols
    )[["PC1", "PC2"]]

cum_var_fig = build_cumulative_variance_plot(pca_step)
biplot_fig = build_biplot(scores_df, loading_df, scores_df["cluster"], config["segment_label"])
pair_fig = build_pair_scatter(scores_df, scores_df["cluster"], config["segment_label"])

st.subheader("PCA View")
st.plotly_chart(cum_var_fig, use_container_width=True)
st.plotly_chart(biplot_fig, use_container_width=True)
st.pyplot(pair_fig, clear_figure=True)

# Feature distributions by cluster
clustered_features = X_model.copy()
clustered_features["cluster"] = labels
st.subheader("What separates the clusters?")
standardize_boxes = choice == "Pitchers"
st.plotly_chart(
    build_feature_boxplots(clustered_features, config["segment_label"], standardize=standardize_boxes),
    use_container_width=True,
)
if standardize_boxes:
    st.caption(
        "Pitcher feature boxplots are z-scored to neutralize scale differences (e.g., BB_rate vs HR_rate) "
        "so cluster separation reflects shape rather than magnitude."
    )

# Salary/contract comparisons
salary_df = Xorig.copy()
salary_df["cluster"] = labels
if {"avg_salary_year", "contract_length"}.issubset(salary_df.columns):
    st.subheader("Salary and Contract Length Comparisons")
    salary_fig, contract_fig = build_salary_contract_plots(salary_df, config["segment_label"])
    st.plotly_chart(salary_fig, use_container_width=True)
    st.plotly_chart(contract_fig, use_container_width=True)
    t_results = run_t_tests(salary_df)
    if t_results:
        st.subheader("T-Tests: Salary and Contract Differences")
        for label, t_stat, p_val in t_results:
            st.write(f"{label}: t = {t_stat:.4f}, p = {p_val:.4e}")
else:
    st.info("Salary/contract fields were not found in this dataset.")
