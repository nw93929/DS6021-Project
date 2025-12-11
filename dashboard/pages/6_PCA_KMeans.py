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
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="K-Means Segmentation", page_icon="🎯", layout="wide")

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


# --- NEW GLOBAL HELPER FUNCTIONS START HERE ---

def cohens_d(x, y):
    """Calculates Cohen's d for two independent samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    # Check for zero degrees of freedom to prevent division by zero in variance pool
    if dof <= 0: return 0.0
    
    # Calculate pooled standard deviation
    std_pool = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
    if std_pool == 0: return 0.0
    
    return (np.mean(x) - np.mean(y)) / std_pool


def get_effect_size_label(value, test_type):
    """Provides a qualitative label for the quantitative effect size (Cohen's d or Eta-Squared)."""
    if test_type == "T-test":
        # Cohen's d standards
        value = abs(value)
        if value < 0.2: return "Negligible"
        elif value < 0.5: return "Small"
        elif value < 0.8: return "Medium"
        else: return "Large"
    elif test_type == "ANOVA":
        # Cohen's standard for Eta-Squared (small: 0.01, medium: 0.06, large: 0.14)
        if value < 0.01: return "Negligible"
        elif value < 0.06: return "Small"
        elif value < 0.14: return "Medium"
        else: return "Large"
    return "N/A"

# --- NEW GLOBAL HELPER FUNCTIONS END HERE ---


# Feature preparation functions remain the same
def prepare_batter_features(df: pd.DataFrame):
    X = df.select_dtypes(include=[np.number]).copy()
    drop_cols = ["year", "free_agent_salary", "PB", "WP", "won_cy_young", "won_mvp", "won_gold_glove", "won_silver_slugger", "all_star", "PO", "InnOuts", "A", "E", "ZR"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    X["PA"] = X["AB"] + X["BB"] + X["HBP"] + X["SF"] + X["SH"]
    X["SLG"] = safe_divide((X["H"] - X["2B"] - X["3B"] - X["HR"]) + (2 * X["2B"]) + (3 * X["3B"]) + (4 * X["HR"]), X["AB"])
    X["BB_rate"] = safe_divide(X["BB"], X["PA"])
    X["SO_rate"] = safe_divide(X["SO"], X["PA"])
    X["BABIP"] = safe_divide(X["H"] - X["HR"], X["AB"] - X["SO"] - X["HR"] + X["SF"])
    X["SB_EFF"] = safe_divide(X["SB"], X["SB"] + X["CS"])
    count_cols = ["AB", "PA", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP", "DP"]
    X.drop(columns=[c for c in count_cols if c in X.columns], inplace=True)
    missing_snapshot = X.copy()
    X = X.dropna()
    Xorig = X.copy()
    X_model = X.drop(columns=[c for c in ["avg_salary_year", "contract_length"] if c in X.columns])
    return X_model, Xorig, missing_snapshot


def prepare_pitcher_features(df: pd.DataFrame):
    X = df.select_dtypes(include=[np.number]).copy()
    drop_cols = ["year", "free_agent_salary", "won_cy_young", "won_mvp", "won_gold_glove", "won_silver_slugger", "all_star", "PO", "A", "E", "DP", "PB", "ZR", "WP", "WP.1", "BK"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    X["IP"] = safe_divide(X["InnOuts"], 3)
    X["K_rate"] = safe_divide(X["SO"], X["BFP"])
    X["BB_rate"] = safe_divide(X["BB"], X["BFP"])
    X["HR_rate"] = safe_divide(X["HR"], X["BFP"])
    X["GS_rate"] = safe_divide(X["GS"], X["G"])
    X["WHIP"] = safe_divide(X["H"] + X["BB"], X["IP"])
    count_cols = ["W", "L", "G", "GS", "CG", "SHO", "SV", "H", "ER", "HR", "BB", "SO", "IBB", "BFP", "GF", "R", "SH", "SF", "GIDP", "InnOuts", "IP"]
    X.drop(columns=[c for c in count_cols if c in X.columns], inplace=True)
    missing_snapshot = X.copy()
    X = X.dropna()
    Xorig = X.copy()
    X_model = X.drop(columns=[c for c in ["avg_salary_year", "contract_length"] if c in X.columns])
    return X_model, Xorig, missing_snapshot

# Plotting functions remain the same
def build_missing_heatmap(df: pd.DataFrame):
    fig = px.imshow(df.isnull(), aspect="auto", color_continuous_scale="Greys", title="Missing Values (pre-dropna)")
    fig.update_layout(height=400, xaxis_title="Feature", yaxis_title="Row")
    return fig


def build_elbow_plot(k_values, wcss):
    fig = px.line(x=k_values, y=wcss, markers=True, labels={"x": "Number of Clusters (k)", "y": "WCSS / Inertia"}, title="Elbow Plot (StandardScaler → K-Means)")
    fig.update_layout(height=450)
    return fig


def build_silhouette_plot(k_values, sil_scores):
    fig = px.line(x=k_values, y=sil_scores, markers=True, labels={"x": "Number of Clusters (k)", "y": "Silhouette Score"}, title="Silhouette Scores by k")
    fig.update_layout(height=450)
    return fig


def build_cumulative_variance_plot(pca: PCA):
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    fig = px.line(x=np.arange(1, len(cum) + 1), y=cum, markers=True, labels={"x": "Number of Components", "y": "Cumulative Explained Variance"}, title="Cumulative Explained Variance by Component")
    fig.add_shape(type="line", x0=1, y0=0.85, x1=len(cum), y1=0.85, line=dict(color="Red", width=1, dash="dash"))
    max_comp = len(cum)
    fig.add_annotation(x=max_comp, y=0.85, text="85% Threshold", showarrow=False, xanchor='right', yanchor='bottom', font=dict(color="Red", size=10))
    return fig


def build_biplot_interactive(scores_df: pd.DataFrame, loading_df: pd.DataFrame, clusters: pd.Series, segment_label: str, pc_x: int, pc_y: int, scale_factor: float, cluster_distinction: bool,):
    pc_x_col = f"PC{pc_x}"
    pc_y_col = f"PC{pc_y}"
    if pc_x_col not in scores_df.columns or pc_y_col not in scores_df.columns:
        st.error(f"Cannot plot {pc_x_col} or {pc_y_col}. Max PC available: {scores_df.shape[1]-1}")
        return go.Figure()

    fig = go.Figure()
    if cluster_distinction:
        for cluster_id in sorted(clusters.unique()):
            cluster_mask = clusters == cluster_id
            fig.add_trace(go.Scatter(x=scores_df.loc[cluster_mask, pc_x_col], y=scores_df.loc[cluster_mask, pc_y_col], mode="markers", marker=dict(size=8, opacity=0.7), name=f"Cluster {cluster_id}"))
    else:
        fig.add_trace(go.Scatter(x=scores_df[pc_x_col], y=scores_df[pc_y_col], mode="markers", marker=dict(size=8, opacity=0.7), name="All Data Points"))

    loading_df_filtered = loading_df.copy()
    loading_df_filtered['var'] = loading_df_filtered.index
    for var_name, row in loading_df_filtered.iterrows():
        pc_x_load = row[pc_x_col] if pc_x_col in row else 0
        pc_y_load = row[pc_y_col] if pc_y_col in row else 0
        fig.add_trace(go.Scatter(x=[0, pc_x_load * scale_factor], y=[0, pc_y_load * scale_factor], mode="lines+markers+text", text=[None, var_name], textposition="top center", line=dict(color="crimson", width=2), marker=dict(size=5, color="crimson"), showlegend=False, textfont=dict(size=12)))

    fig.update_layout(
        title=f"PCA Biplot: {pc_x_col} vs {pc_y_col} — {segment_label}",
        xaxis_title=pc_x_col, yaxis_title=pc_y_col, height=650,
        xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey'),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey'),
    )
    return fig


def build_pair_scatter(df: pd.DataFrame, clusters: pd.Series, segment_label: str, data_type: str):
    """Generates a pairplot for either PCA Scores (all available PCs) or Original Standardized Features."""
    
    plot_df = df.copy()
    plot_df["cluster"] = clusters.astype(str)

    if data_type == "PCA Scores":
        dims = [c for c in plot_df.columns if c.startswith("PC")]
        title_suffix = " (All PCA Scores)"
        plot_df = plot_df[['cluster'] + dims]
    else: # Original Data
        feature_cols = [c for c in plot_df.columns if c not in ('cluster', 'avg_salary_year', 'contract_length')]
        plot_df = plot_df[['cluster'] + feature_cols]
        scaler = StandardScaler()
        plot_df[feature_cols] = scaler.fit_transform(plot_df[feature_cols])
        if len(feature_cols) > 8:
            feature_cols = ['age', 'SLG', 'BB_rate', 'SO_rate', 'BABIP', 'SB_EFF'] if segment_label == "Batter Segment" else ['age', 'ERA', 'BAOpp', 'K_rate', 'GS_rate', 'WHIP']
            plot_df = plot_df[['cluster'] + feature_cols]
        title_suffix = " (Original Standardized Features)"

    sns.set(style="ticks")
    vars_to_plot = plot_df.columns.drop('cluster', errors='ignore').tolist()
    if not vars_to_plot:
        st.warning("No valid features found for pair plot.")
        return plt.figure()
        
    pair_grid = sns.pairplot(plot_df, hue="cluster", diag_kind="kde", palette="Set2", plot_kws={"alpha": 0.7, "s": 40}, vars=vars_to_plot)
    pair_grid.fig.suptitle(f"Pairwise Scatter Plots — {segment_label}{title_suffix}", y=1.02)
    plt.tight_layout()
    return pair_grid.fig


def build_feature_distribution_plot(X_with_clusters: pd.DataFrame, segment_label: str, chart_type: str):
    """Builds distribution plots (Box, Violin) of features on a standardized (Z-scored) scale."""
    features_df = X_with_clusters.copy()
    feature_cols = [c for c in features_df.columns if c != "cluster"]
    
    scaler = StandardScaler()
    features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])
    y_label = "Z-Scored Feature Value"
    title_suffix = f" (Standardized Scale, {chart_type})"

    plot_df = features_df.melt(id_vars="cluster", value_vars=feature_cols, var_name="Feature", value_name="Value")
    
    base_params = {
        "data_frame": plot_df,
        "x": "Feature",
        "y": "Value",
        "color": "cluster",
        "title": f"Feature Distributions by Cluster — {segment_label}{title_suffix}",
        "color_discrete_sequence": px.colors.qualitative.Set2,
    }

    if chart_type == "Boxplot":
        fig = px.box(**base_params)
    elif chart_type == "Violin Plot":
        fig = px.violin(**base_params)
    else:
        fig = px.box(**base_params)

    fig.update_layout(xaxis_tickangle=-45, height=600, yaxis_title=y_label)
    return fig


def build_compensation_plot(df_with_clusters: pd.DataFrame, segment_label: str, metric: str, chart_type: str):
    """Builds an interactive box/violin plot for compensation metrics."""
    
    metric_map = {
        "avg_salary_year": {"label": "Avg Salary per Year ($)", "format": "$,.2s", "title": "Average Yearly Salary"},
        "contract_length": {"label": "Years", "format": None, "title": "Contract Length"},
    }
    
    params = metric_map[metric]
    
    base_params = {
        "data_frame": df_with_clusters,
        "x": "cluster",
        "y": metric,
        "color": "cluster",
        "title": f"{params['title']} by Cluster — {segment_label}",
        "labels": {metric: params['label'], "cluster": segment_label},
        "color_discrete_sequence": px.colors.qualitative.Set2,
    }

    if chart_type == "Boxplot":
        fig = px.box(**base_params)
    elif chart_type == "Violin Plot":
        fig = px.violin(**base_params)
    else:
        fig = px.box(**base_params)
        
    if params['format']:
        fig.update_yaxes(tickformat=params['format'])
        
    return fig


def run_compensation_tests(df_with_clusters: pd.DataFrame, k_selected: int):
    
    if not {"avg_salary_year", "contract_length", "cluster"}.issubset(df_with_clusters.columns):
        return None, None
    
    results = []
    tukey_results = {}
    
    # Ensure 'cluster' is treated as categorical for statsmodels
    df_for_stats = df_with_clusters.copy().dropna(subset=["avg_salary_year", "contract_length"])
    if df_for_stats.empty:
        return None, None
        
    df_for_stats["cluster_cat"] = df_for_stats["cluster"].astype('category')

    if k_selected == 2:
        # --- T-test for k=2 ---
        
        # Salary Comparison
        salary0 = df_for_stats[df_for_stats["cluster"] == "0"]["avg_salary_year"]
        salary1 = df_for_stats[df_for_stats["cluster"] == "1"]["avg_salary_year"]
        t_stat, p_val = stats.ttest_ind(salary0, salary1, equal_var=False)
        d_val = cohens_d(salary0, salary1)
        # Store results: (Metric, T-stat, P-value, Cohen's d, TestType, df_num, Sig_Status)
        results.append(("Average Salary", t_stat, p_val, d_val, "T-test", None, p_val < 0.05))

        # Contract Length Comparison
        c0 = df_for_stats[df_for_stats["cluster"] == "0"]["contract_length"]
        c1 = df_for_stats[df_for_stats["cluster"] == "1"]["contract_length"]
        t_stat_c, p_val_c = stats.ttest_ind(c0, c1, equal_var=False)
        d_val_c = cohens_d(c0, c1)
        results.append(("Contract Length", t_stat_c, p_val_c, d_val_c, "T-test", None, p_val_c < 0.05))
        
    elif k_selected > 2:
        # --- ANOVA & Tukey for k>2 ---
        
        metrics = ["avg_salary_year", "contract_length"]
        for metric in metrics:
            
            # 1. Run ANOVA
            formula = f'{metric} ~ cluster_cat'
            lm = ols(formula, data=df_for_stats).fit()
            aov = anova_lm(lm)
            
            f_stat = aov.loc["cluster_cat", "F"]
            p_val = aov.loc["cluster_cat", "PR(>F)"]
            ss_between = aov.loc["cluster_cat", "sum_sq"]
            ss_total = aov.loc["cluster_cat", "sum_sq"] + aov.loc["Residual", "sum_sq"]
            eta_squared = safe_divide(ss_between, ss_total)
            
            # Store ANOVA results: (Metric, F-stat, P-value, Eta-Squared, TestType, df_num, Sig_Status)
            is_sig_anova = p_val < 0.05
            results.append((
                "Average Salary" if metric == "avg_salary_year" else "Contract Length",
                f_stat, 
                p_val, 
                eta_squared, 
                "ANOVA", 
                aov.loc["cluster_cat", "df"],
                is_sig_anova
            ))
            
            # 2. Run Tukey's HSD if ANOVA is significant
            if is_sig_anova:
                tukey_hsd = pairwise_tukeyhsd(endog=df_for_stats[metric], groups=df_for_stats["cluster_cat"], alpha=0.05)
                # Convert results to a pandas DataFrame for better presentation
                tukey_df = pd.DataFrame(data=tukey_hsd._results_table.data[1:], columns=tukey_hsd._results_table.data[0])
                tukey_df = tukey_df[['group1', 'group2', 'meandiff', 'lower', 'upper', 'reject']]
                tukey_df.columns = ['Cluster 1', 'Cluster 2', 'Mean Diff', 'Lower CI', 'Upper CI', 'Significant']
                tukey_results[metric] = tukey_df
                
    return results, tukey_results

# --- CORE LOGIC TO RUN MODEL AND GET RESULTS ---

@st.cache_data(show_spinner="Running K-Means and PCA for selected cluster size...")
def run_pca_kmeans(X_model: pd.DataFrame, X_config: dict, k: int):
    # 1. Run Pipeline
    final_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=X_config["n_components"], random_state=RANDOM_STATE)),
            ("kmeans", KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=RANDOM_STATE)),
        ]
    )
    final_pipe.fit(X_model)
    labels = final_pipe["kmeans"].labels_.astype(str)
    pca_step: PCA = final_pipe["pca"]
    
    # 2. Get Scores
    scores = final_pipe[:-1].transform(X_model)
    pc_cols = [f"PC{i}" for i in range(1, scores.shape[1] + 1)]
    scores_df = pd.DataFrame(scores, columns=pc_cols, index=X_model.index)
    scores_df["cluster"] = labels

    # 3. Get Loadings
    loading_df = pd.DataFrame(
        pca_step.components_.T, index=X_model.columns, columns=pc_cols
    )
    return labels, pca_step, scores_df, loading_df

# --- STREAMLIT APP ---

st.title("K-Means Segmentation")
st.markdown(
    """
Unsupervised clustering on MLB player performance using a StandardScaler → PCA → K-Means pipeline.
Use the interactive controls below to experiment with cluster size ($k$) and visualization parameters.
"""
)

choice = st.selectbox("Choose which group to analyze:", list(DATA_CONFIG.keys()))
config = DATA_CONFIG[choice]

with st.spinner("Loading data and preparing features..."):
    df = load_dataset(config["file"])
    if choice == "Batters":
        X_model, Xorig, missing_snapshot = prepare_batter_features(df)
    else:
        X_model, Xorig, missing_snapshot = prepare_pitcher_features(df)

st.write(f"Rows after cleaning: **{len(X_model)}**, features used: **{X_model.shape[1]}**")

# --- CLUSTERING DIAGNOSTICS (Static, always run for up to k=10) ---
with st.spinner("Running clustering diagnostics..."):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_model)

    k_values = list(range(1, 11))
    wcss = []
    sil_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=RANDOM_STATE)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        if k >= 2:
            sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    # Determine optimal k based on max silhouette score
    sil_k = list(range(2, 11))
    if sil_scores:
        optimal_k = sil_k[np.argmax(sil_scores)]
        optimal_score = np.max(sil_scores)
        k_options_map = {f"Optimal: k={optimal_k} (Silhouette={optimal_score:.3f})": optimal_k}
    else:
        optimal_k = 2
        k_options_map = {f"k={optimal_k} (Default)": optimal_k}

    # Add other k options
    for k in sil_k:
        if k != optimal_k:
            k_options_map[f"k={k}"] = k
    
    sorted_k_options = sorted(k_options_map.keys(), key=lambda x: k_options_map[x])
    default_k_index = sorted_k_options.index([key for key, val in k_options_map.items() if val == optimal_k][0])


st.header("⚙️ Model Configuration")
col_k_select, col_empty = st.columns([1, 2])
with col_k_select:
    k_choice_label = st.selectbox(
        "Select Number of Clusters ($k$):",
        options=sorted_k_options,
        index=default_k_index,
        key='k_selector'
    )
    k_selected = k_options_map[k_choice_label]

# Run the model with the selected k
labels, pca_step, scores_df, loading_df = run_pca_kmeans(X_model, config, k_selected)

# --- RESULTS VISUALIZATIONS ---

st.subheader("Clustering Diagnostics")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(build_elbow_plot(k_values, wcss), use_container_width=True)
with col2:
    st.plotly_chart(build_silhouette_plot(sil_k, sil_scores), use_container_width=True)


# 1. Interactive Biplot
st.header("🔬 Interactive PCA Biplot")
st.markdown("Use this visualization to explore how the selected clusters separate in the principal component space, and which features drive that separation.")
st.plotly_chart(build_cumulative_variance_plot(pca_step), use_container_width=True)

# Collect user inputs for the biplot
max_pc = config["n_components"]
pc_options = list(range(1, max_pc + 1))
default_pc_y = 2 if max_pc >= 2 else 1

col_pc_x, col_pc_y, col_scale = st.columns([1, 1, 1.5])
with col_pc_x:
    pc_x_input = st.selectbox("Select X-axis PC:", options=pc_options, index=pc_options.index(1), key='pc_x_biplot')
with col_pc_y:
    pc_y_input = st.selectbox("Select Y-axis PC:", options=pc_options, index=pc_options.index(default_pc_y), key='pc_y_biplot')
with col_scale:
    scale_factor_input = st.slider("Arrow Scale Factor:", 1.0, 15.0, 8.0, 0.5)
    cluster_distinction_input = st.checkbox("Show Cluster Distinction", value=True)

# Generate and display the interactive biplot
interactive_biplot_fig = build_biplot_interactive(
    scores_df, loading_df, scores_df["cluster"], config["segment_label"],
    pc_x=pc_x_input, pc_y=pc_y_input, 
    scale_factor=scale_factor_input, cluster_distinction=cluster_distinction_input
)
st.plotly_chart(interactive_biplot_fig, use_container_width=True)

# PCA Loadings Table
st.subheader("Principal Component Loadings (Feature Weights)")
st.markdown(
    "These values indicate how strongly each original feature contributes to the variance captured by the Principal Components (PCs). "
    "Features with large absolute values primarily define the PC."
)

top_pc_cols = [f"PC{i}" for i in range(1, config['n_components'] + 1)]
st.dataframe(
    loading_df[top_pc_cols].style.background_gradient(cmap='RdYlGn', axis=1, subset=top_pc_cols, vmin=-1.0, vmax=1.0),
    use_container_width=True
)


# 2. Interactive Pairwise Scatter Plot
st.header("🔗 Interactive Pairwise Scatter Plots")
st.markdown("Visualize cluster separation using the reduced PCA scores (all components) or the original standardized features.")
pair_data_choice = st.radio(
    "Select data for pairwise plots:",
    ("PCA Scores", "Original Standardized Features"),
    horizontal=True,
)

# Prepare data based on choice
if pair_data_choice == "PCA Scores":
    pair_data_df = scores_df.drop(columns='cluster')
else:
    pair_data_df = X_model.copy()
    
# Generate and display the interactive pairplot
pair_fig = build_pair_scatter(pair_data_df, scores_df["cluster"], config["segment_label"], pair_data_choice)
st.pyplot(pair_fig, clear_figure=True)


# Feature distributions 
clustered_features = X_model.copy()
clustered_features["cluster"] = labels
st.header("📊 Interactive Feature Distributions by Cluster")

# Chart Type Selector for feature distributions
chart_type_input_features = st.selectbox(
    "Select Chart Type (Features):",
    ("Boxplot", "Violin Plot"),
    index=1,
    key='dist_chart_type_features'
)

st.caption("All features are shown on a Z-scored (Standardized) scale for direct comparison of means, variance, and distribution shape.")
st.plotly_chart(
    build_feature_distribution_plot(clustered_features, config["segment_label"], chart_type_input_features),
    use_container_width=True,
)


# Compensation comparisons
salary_df = Xorig.copy()
salary_df["cluster"] = labels
if {"avg_salary_year", "contract_length"}.issubset(salary_df.columns):
    st.header("💵 Compensation Analysis")
    st.markdown(f"Compare compensation metrics between the **{k_selected}** discovered clusters.")
    
    # Chart Type Selector for compensation plots
    chart_type_input_compensation = st.radio(
        "Select Chart Type (Compensation):",
        ("Boxplot", "Violin Plot"),
        index=0,
        key='dist_chart_type_compensation',
        horizontal=True
    )
    
    col_salary, col_contract = st.columns(2)
    with col_salary:
        salary_fig = build_compensation_plot(salary_df, config["segment_label"], "avg_salary_year", chart_type_input_compensation)
        st.plotly_chart(salary_fig, use_container_width=True)
    with col_contract:
        contract_fig = build_compensation_plot(salary_df, config["segment_label"], "contract_length", chart_type_input_compensation)
        st.plotly_chart(contract_fig, use_container_width=True)
    
    # --- Compensation Statistical Tests ---
    comp_results, tukey_results = run_compensation_tests(salary_df, k_selected)
    
    if comp_results:
        st.subheader("Statistical Tests on Compensation")
        
        if k_selected == 2:
            st.markdown("Using **T-test** to compare the means of the two independent groups. Cohen's $d$ measures the standardized difference.")
            
            test_data = []
            for label, t_stat, p_val, d_val, _, _, _ in comp_results:
                is_sig = p_val < 0.05
                sig_text = "**Significant**" if is_sig else "Not Significant"
                # CORRECTED CALL: Using the now global get_effect_size_label
                effect_size_label = get_effect_size_label(d_val, "T-test")
                
                test_data.append([
                    label, 
                    f"$t = {t_stat:.4f}$", 
                    f"${p_val:.4e}$", 
                    sig_text,
                    f"${d_val:.4f}$ ({effect_size_label})"
                ])

            st.table(pd.DataFrame(test_data, columns=["Metric", "Test Statistic (t)", "P-value", "Significance", "Cohen's d (Effect Size)"]))

        elif k_selected > 2:
            st.markdown(r"Using **ANOVA F-test** to determine if *any* cluster mean is different. $\eta^2$ (Eta-Squared) measures the proportion of variance explained by the cluster grouping.")
            
            test_data = []
            for label, f_stat, p_val, eta_squared, _, df_num, is_sig_anova in comp_results:
                is_sig_anova = p_val < 0.05
                sig_text = "**Significant**" if is_sig_anova else "Not Significant"
                # CORRECTED CALL: Using the now global get_effect_size_label
                effect_size_label = get_effect_size_label(eta_squared, "ANOVA")
                
                test_data.append([
                    label, 
                    f"$F({df_num:.0f}) = {f_stat:.4f}$", 
                    f"${p_val:.4e}$", 
                    sig_text,
                    f"${eta_squared:.4f}$ ({effect_size_label})"
                ])
                
            st.table(pd.DataFrame(test_data, columns=["Metric", "Test Statistic (F)", "P-value", "Significance", "Eta-Squared ($\eta^2$ Effect Size)"]))
            
            # --- Tukey's HSD Post-hoc Test ---
            st.subheader(f"Post-hoc Analysis (Tukey's HSD)")
            st.markdown(
                "Tukey's Honestly Significant Difference (HSD) test is performed for metrics where ANOVA detected a significant difference ($p < 0.05$). "
                "This test adjusts for multiple comparisons to determine which specific pairs of clusters are significantly different."
            )
            
            # Use tuple mapping to retrieve the ANOVA significance status easily
            anova_sig_map = {r[0]: r[6] for r in comp_results}
            
            found_tukey_results = False
            for metric, tukey_df in tukey_results.items():
                
                # Check if ANOVA was significant for this metric
                label_key = "Average Salary" if metric == "avg_salary_year" else "Contract Length"
                if anova_sig_map.get(label_key, False):
                    found_tukey_results = True
                    st.markdown(f"#### Results for **{metric.replace('_', ' ').title()}**")
                    
                    # Format the table for display
                    tukey_display_df = tukey_df.copy()
                    tukey_display_df['Mean Diff'] = tukey_display_df['Mean Diff'].apply(lambda x: f'{x:,.2f}')
                    tukey_display_df['Lower CI'] = tukey_display_df['Lower CI'].apply(lambda x: f'{x:,.2f}')
                    tukey_display_df['Upper CI'] = tukey_display_df['Upper CI'].apply(lambda x: f'{x:,.2f}')
                    tukey_display_df['Significant'] = tukey_display_df['Significant'].apply(lambda x: 'Yes' if x else 'No')

                    # Highlight significant rows
                    def highlight_significant(row):
                        if row['Significant'] == 'Yes':
                            return ['background-color: #d4edda'] * len(row)  # Light green
                        return [''] * len(row)
                    
                    st.dataframe(
                        tukey_display_df.style.apply(highlight_significant, axis=1),
                        use_container_width=True
                    )
                
            if not found_tukey_results:
                st.info("No Tukey's HSD test was performed because the overall ANOVA test was not significant ($p > 0.05$) for either compensation metric.")

else:
    st.info("Salary/contract fields were not found in this dataset.")