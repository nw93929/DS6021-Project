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
# NOTE: Ensure this path is correct for your specific project structure
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cleaned"

DATA_CONFIG = {
    "Batters": {
        "file": "final_batters_df.csv",
        "n_components": 5,
        "summary": (
            "**Optimal Clusters (k=2):** Segmentation is primarily based on **Age/Approach**.\n\n"
            "**Cluster 0 (Experienced/Disciplined):** Older players with significantly **lower Strikeout Rates** "
            "and generally lower BB\_rates, suggesting a mature contact/favorable-count approach.\n\n"
            "**Cluster 1 (Younger/Aggressive):** Younger players with significantly **higher Strikeout Rates** "
            "and higher BB\_rates, indicative of a power/high-risk approach.\n\n"
            "**Compensation:** There is a **statistically significant and medium effect size gap in salary** "
            "favoring Cluster 1 (Younger/Aggressive), suggesting the market highly values their power profile "
            "despite the higher strikeout risk. Contract length differences are negligible."
        ),
        "segment_label": "Batter Segment",
    },
    "Pitchers": {
        "file": "final_pitchers_df.csv",
        "n_components": 5,
        "summary": (
            "**Optimal Clusters (k=2):** Segmentation is clearly driven by **Pitcher Role**.\n\n"
            "**Cluster 0 (Relievers/Specialists):** Defined by very low **GS\_rate** (few starts), "
            "higher K\_rates, and higher BB\_rates, typical of short-inning power pitchers.\n\n"
            "**Cluster 1 (Starting Pitchers):** Defined by high **GS\_rate** (many starts), "
            "lower K\_rates, and lower BB\_rates, indicative of pitchers relied on for workload and control.\n\n"
            "**Compensation:** While the role distinction is clear, the clustering revealed **negligible differences** "
            "in both average salary and contract length between the Starter and Reliever segments, suggesting the market "
            "values the two roles similarly on a per-player basis, or that other factors dominate salary."
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

    # Rate-based features to mirror notebook
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
    fig.add_shape(
        type="line",
        x0=1, y0=0.85, x1=len(cum), y1=0.85,
        line=dict(color="Red", width=1, dash="dash"),
    )
    fig.add_annotation(
        x=len(cum) + 0.1, y=0.85,
        text="85% Threshold",
        showarrow=False,
        font=dict(color="Red"),
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
                textfont=dict(size=12)
            )
        )

    fig.update_layout(
        title=f"PCA Biplot (PC1 vs PC2) with Loadings — {segment_label}",
        xaxis_title="PC1",
        yaxis_title="PC2",
        height=650,
        # Ensure origin (0,0) is visible for the loading vectors
        xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey'),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey'),
    )
    return fig


def build_pair_scatter(scores_df: pd.DataFrame, clusters, segment_label: str):
    # Use PC1, PC2, PC3 for the pair plot
    dims = [c for c in scores_df.columns if c.startswith("PC")][:3]
    plot_df = scores_df[dims].copy()
    plot_df["cluster"] = clusters.astype(str)
    sns.set(style="ticks")
    
    # Create the seaborn pairplot
    pair_grid = sns.pairplot(
        plot_df,
        hue="cluster",
        diag_kind="kde",
        palette="Set2",
        plot_kws={"alpha": 0.7, "s": 40},
    )
    pair_grid.fig.suptitle(f"Pairwise PCA Scores with KDE Diagonals — {segment_label}", y=1.02)
    plt.tight_layout() # Added for better layout in Streamlit
    return pair_grid.fig


def build_feature_boxplots(X_with_clusters: pd.DataFrame, segment_label: str, standardize: bool = False):
    features_df = X_with_clusters.copy()
    feature_cols = [c for c in features_df.columns if c != "cluster"]
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])
        y_label = "Z-Scored Feature Value"
        title_suffix = " (Standardized)"
    else:
        y_label = "Feature Value"
        title_suffix = " (Original Scale)"

    plot_df = features_df.melt(
        id_vars="cluster",
        value_vars=feature_cols,
        var_name="Feature",
        value_name="Value",
    )
    fig = px.box(
        plot_df,
        x="Feature",
        y="Value",
        color="cluster",
        title=f"Feature Distributions by Cluster — {segment_label}{title_suffix}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(xaxis_tickangle=-45, height=600, yaxis_title=y_label)
    return fig


def build_salary_contract_plots(df_with_clusters: pd.DataFrame, segment_label: str):
    salary_fig = px.box(
        df_with_clusters,
        x="cluster",
        y="avg_salary_year",
        color="cluster",
        title=f"Average Yearly Salary by Cluster — {segment_label}",
        labels={"avg_salary_year": "Avg Salary per Year ($)", "cluster": segment_label},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    salary_fig.update_yaxes(tickformat="$,.2s")

    contract_fig = px.box(
        df_with_clusters,
        x="cluster",
        y="contract_length",
        color="cluster",
        title=f"Contract Length by Cluster — {segment_label}",
        labels={"contract_length": "Years", "cluster": segment_label},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return salary_fig, contract_fig


def run_t_tests(df_with_clusters: pd.DataFrame):
    results = []
    
    # Cohen's d helper
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        std_pool = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
        # Handle case where pooled standard deviation is near zero
        if std_pool == 0:
            return 0
        return (np.mean(x) - np.mean(y)) / std_pool

    if {"avg_salary_year", "contract_length", "cluster"}.issubset(df_with_clusters.columns):
        # Salary Comparison
        salary0 = df_with_clusters[df_with_clusters["cluster"] == "0"]["avg_salary_year"]
        salary1 = df_with_clusters[df_with_clusters["cluster"] == "1"]["avg_salary_year"]
        t_stat, p_val = stats.ttest_ind(salary0, salary1, equal_var=False, nan_policy='omit')
        d_val = cohens_d(salary0.dropna(), salary1.dropna())
        results.append(("Average Salary", t_stat, p_val, d_val))

        # Contract Length Comparison
        c0 = df_with_clusters[df_with_clusters["cluster"] == "0"]["contract_length"].dropna()
        c1 = df_with_clusters[df_with_clusters["cluster"] == "1"]["contract_length"].dropna()
        t_stat_c, p_val_c = stats.ttest_ind(c0, c1, equal_var=False)
        d_val_c = cohens_d(c0, c1)
        results.append(("Contract Length", t_stat_c, p_val_c, d_val_c))
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

st.header("📋 Analysis Summary")
st.markdown(config["summary"])

with st.spinner("Loading data and preparing features..."):
    df = load_dataset(config["file"])
    if choice == "Batters":
        X_model, Xorig, missing_snapshot = prepare_batter_features(df)
    else:
        X_model, Xorig, missing_snapshot = prepare_pitcher_features(df)

st.write(f"Rows after cleaning: **{len(X_model)}**, features used: **{X_model.shape[1]}**")

# Missingness view
st.subheader("Missing Value Check")
st.plotly_chart(build_missing_heatmap(missing_snapshot), use_container_width=True)

# Elbow and silhouette diagnostics
st.subheader("Clustering Diagnostics")
with st.spinner("Running clustering diagnostics..."):
    base_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            # PCA here is mostly to speed up calculation and ensure consistent scaling before K-Means
            ("pca", PCA(n_components=config["n_components"], random_state=RANDOM_STATE)), 
            ("kmeans", KMeans(init="k-means++", n_init="auto", random_state=RANDOM_STATE)),
        ]
    )

    k_values = list(range(1, 11))
    wcss = []
    for k in k_values:
        # NOTE: Using PCA to reduce noise and speed up K-Means fit for diagnostics
        pipe_for_wcss = Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=RANDOM_STATE))])
        pipe_for_wcss.fit(X_model)
        wcss.append(pipe_for_wcss["kmeans"].inertia_)

    sil_scores = []
    sil_k = list(range(2, 11))
    for k in sil_k:
        pipe_for_sil = Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=RANDOM_STATE))])
        pipe_for_sil.fit(X_model)
        sil_scores.append(silhouette_score(StandardScaler().fit_transform(X_model), pipe_for_sil["kmeans"].labels_))

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

st.subheader("PCA Dimensionality Reduction")
st.plotly_chart(cum_var_fig, use_container_width=True)

st.subheader("Principal Components Analysis")
st.plotly_chart(biplot_fig, use_container_width=True)

# Pairplot (using matplotlib/seaborn requires st.pyplot)
pair_fig = build_pair_scatter(scores_df, scores_df["cluster"], config["segment_label"])
st.pyplot(pair_fig, clear_figure=True)

# Feature distributions by cluster
clustered_features = X_model.copy()
clustered_features["cluster"] = labels
st.subheader("Feature Distributions by Cluster")
# Pass the boolean flag to standardize only for Pitchers
standardize_boxes = choice == "Pitchers"
st.plotly_chart(
    build_feature_boxplots(clustered_features, config["segment_label"], standardize=standardize_boxes),
    use_container_width=True,
)
if standardize_boxes:
    st.caption(
        "Pitcher feature boxplots are Z-Scored to equalize scale (e.g., K\_rate vs HR\_rate) for better visualization of cluster separation."
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
        st.subheader("Statistical Tests on Compensation")
        test_data = []
        for label, t_stat, p_val, d_val in t_results:
            is_sig = p_val < 0.05
            sig_text = "Significant" if is_sig else "Not Significant"
            effect_size = ""
            if abs(d_val) < 0.2: effect_size = "Negligible"
            elif abs(d_val) < 0.5: effect_size = "Small"
            elif abs(d_val) < 0.8: effect_size = "Medium"
            else: effect_size = "Large"
            
            test_data.append([
                label, 
                f"{t_stat:.4f}", 
                f"{p_val:.4e}", 
                sig_text,
                f"{d_val:.4f} ({effect_size})"
            ])

        st.table(pd.DataFrame(test_data, columns=["Metric", "T-statistic", "P-value", "Significance", "Cohen's d (Effect Size)"]))
else:
    st.info("Salary/contract fields were not found in this dataset.")