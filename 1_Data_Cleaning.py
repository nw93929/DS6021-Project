from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Cleaning", page_icon="🧹")

FINAL_DATASETS: list[tuple[str, tuple[str, ...]]] = [
    ("Final Batters (cleaned)", ("data/cleaned/final_batters.csv", "data/cleaned/final_batters_df.csv")),
    ("Final Pitchers (cleaned)", ("data/cleaned/final_pitchers.csv", "data/cleaned/final_pitchers_df.csv")),
]

PREVIEW_ANCHORS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    # (parent_heading, heading) -> datasets
    "mlb stats api free agent data|data description": [
        ("Free Agents (MLB Stats API)", ("data/raw_data/MLB_stats_free_agents.csv",)),
    ],
    "lahman dataset|data description": [
        ("Lahman Batting", ("data/raw_data/lahman_data/batting_stats.csv",)),
        ("Lahman Pitching", ("data/raw_data/lahman_data/Pitching.csv",)),
        ("Lahman Fielding", ("data/raw_data/lahman_data/Fielding.csv",)),
        ("Lahman Salaries", ("data/raw_data/lahman_data/Salaries.csv",)),
        ("Lahman Awards", ("data/raw_data/lahman_data/awards.csv",)),
        ("Lahman People", ("data/raw_data/lahman_data/People.csv",)),
        ("Lahman Allstar", ("data/raw_data/lahman_data/Allstar.csv",)),
    ],
}

FEATURE_VARIABLES_MD = """
### Data Description (Engineered Feature Set)
- **free_agent_salary** — Average annual salary derived from available Lahman salary records after contract signing.
- **free_agent_salary_log** — `log1p` transform of salary to stabilize variance for linear modeling.
- **contract_length** — Parsed number of contract years from MLB Stats API notes; missing if not found.
- **age_at_fa** — Player age in the free-agent offseason (birth year from Lahman People).
- **primary_pos** — Most-played fielding position over the 3-year window (fallback placeholder if absent).
- **batting_three_year_totals** — Summed counting stats (R, RBI, HR, SB, BB, SO, etc.) across the prior 3 seasons.
- **pitching_three_year_totals** — Summed counting stats (IPOuts, SO, BB, HR, etc.) and averaged rate stats (ERA, BAOpp).
- **fielding_three_year_totals** — Summed putouts/assists/errors plus averaged Zone Rating (ZR).
- **awards_indicators** — Binary flags for Cy Young, MVP, Gold Glove, Silver Slugger, and All-Star in the 3-year window.
- **league_avg_salary** — Mean MLB salary for the free-agent season to contextualize contract values.
- **row_id / playerID / year** — Unique identifiers used for joins and modeling splits.
"""


@st.cache_data
def load_markdown_cells(notebook_path: Path, mtime: float) -> list[str]:
    notebook_json = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_cells = []

    for cell in notebook_json.get("cells", []):
        if cell.get("cell_type") == "markdown":
            markdown_cells.append("".join(cell.get("source", [])))

    return markdown_cells


@st.cache_data
def load_dataset_preview(path_options: tuple[str, ...], nrows: int = 5) -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parents[2]

    for relative_path in path_options:
        path = repo_root / relative_path
        if path.exists():
            return pd.read_csv(path).head(nrows)

    st.warning(f"Dataset not found for options: {', '.join(path_options)}")
    return pd.DataFrame()


def render_markdown_sections(sections: list[str]) -> None:
    if not sections:
        st.warning("No markdown content found in data.ipynb.")
        return

    for section in sections:
        blocks = split_markdown_into_heading_blocks(section)
        current_h1: str | None = None

        for heading_line, level, block in blocks:
            if heading_line and level == 1:
                current_h1 = heading_line

            st.markdown(block)
            maybe_render_feature_description(heading_line, current_h1)
            render_previews_for_heading(heading_line, current_h1)

    # After all markdown content, show the final cleaned dataset snippets
    st.markdown("---")
    st.subheader("Cleaned Dataset Previews")
    render_dataset_previews(FINAL_DATASETS)


def split_markdown_into_heading_blocks(section: str) -> list[tuple[str | None, int, str]]:
    lines = section.splitlines()
    blocks: list[tuple[str | None, int, str]] = []
    current_lines: list[str] = []
    current_heading: str | None = None
    current_level: int = 0

    for line in lines:
        if line.startswith("#"):
            if current_lines:
                blocks.append((current_heading, current_level, "\n".join(current_lines)))
            current_heading = line
            current_level = len(line) - len(line.lstrip("#"))
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append((current_heading, current_level, "\n".join(current_lines)))

    return blocks


def render_previews_for_heading(heading: str | None, parent_heading: str | None) -> None:
    if not heading:
        return

    heading_key = normalize_heading(heading)
    parent_key = normalize_heading(parent_heading) if parent_heading else ""
    lookup_key = f"{parent_key}|{heading_key}"

    datasets = PREVIEW_ANCHORS.get(lookup_key)
    if datasets:
        render_dataset_previews(datasets)


def maybe_render_feature_description(heading: str | None, parent_heading: str | None) -> None:
    if not heading:
        return

    heading_key = normalize_heading(heading)
    parent_key = normalize_heading(parent_heading) if parent_heading else ""

    if parent_key == "data cleaning" and heading_key == "high level overview of feature engineering functions":
        st.markdown(FEATURE_VARIABLES_MD)


def normalize_heading(heading: str) -> str:
    normalized = heading.lstrip("#").strip().lower()
    normalized = normalized.replace("—", "-").replace("–", "-")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return normalized.strip()


def render_dataset_previews(datasets: list[tuple[str, tuple[str, ...]]]) -> None:
    for i in range(0, len(datasets), 2):
        cols = st.columns(2)
        for col, dataset in zip(cols, datasets[i : i + 2]):
            label, path_options = dataset
            preview = load_dataset_preview(path_options)
            with col:
                st.caption(f"{label} (first {len(preview)} rows)")
                st.dataframe(preview, use_container_width=True)


def main() -> None:
    notebook_path = Path(__file__).resolve().parents[2] / "data.ipynb"

    if not notebook_path.exists():
        st.error(f"Notebook not found at {notebook_path}")
        return

    st.sidebar.header("Data Cleaning")
    st.title("Data Cleaning")

    sections = load_markdown_cells(notebook_path, notebook_path.stat().st_mtime)
    render_markdown_sections(sections)

if __name__ == "__main__":
    main()
