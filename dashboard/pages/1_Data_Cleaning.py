from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Cleaning", page_icon="🧹")


@st.cache_data
def load_markdown_cells(notebook_path: Path, mtime: float) -> list[str]:
    notebook_json = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_cells = []

    for cell in notebook_json.get("cells", []):
        if cell.get("cell_type") == "markdown":
            markdown_cells.append("".join(cell.get("source", [])))

    return markdown_cells


def render_markdown_sections(sections: list[str]) -> None:
    if not sections:
        st.warning("No markdown content found in data.ipynb.")
        return

    for section in sections:
        st.markdown(section)


def main() -> None:
    notebook_path = Path(__file__).resolve().parents[2] / "data.ipynb"
    root = Path(__file__).resolve().parents[2]
    batters_path = root / "data" / "cleaned" / "final_batters_df.csv"
    pitchers_path = root / "data" / "cleaned" / "final_pitchers_df.csv"

    if not notebook_path.exists():
        st.error(f"Notebook not found at {notebook_path}")
        return

    st.sidebar.header("Data Cleaning")
    st.title("Data Cleaning")

    sections = load_markdown_cells(notebook_path, notebook_path.stat().st_mtime)
    render_markdown_sections(sections)

    st.markdown("## Variable Dictionary")

    batters_dict = [
        ("row_id", "Unique player-season identifier"),
        ("playerID", "Lahman player ID"),
        ("year", "Season year"),
        ("position", "Primary defensive position"),
        ("age", "Player age in the season"),
        ("avg_salary_year", "Average salary for the season"),
        ("free_agent_salary", "Salary agreed in free-agent deal for the season"),
        ("contract_length", "Length of the active contract (years)"),
        ("AB", "At-bats"),
        ("R", "Runs scored"),
        ("H", "Hits"),
        ("2B", "Doubles"),
        ("3B", "Triples"),
        ("HR", "Home runs"),
        ("RBI", "Runs batted in"),
        ("SB", "Stolen bases"),
        ("CS", "Caught stealing"),
        ("BB", "Walks (non-intentional)"),
        ("SO", "Strikeouts"),
        ("IBB", "Intentional walks"),
        ("HBP", "Hit by pitch"),
        ("SH", "Sacrifice bunts"),
        ("SF", "Sacrifice flies"),
        ("GIDP", "Grounded into double play"),
        ("InnOuts", "Defensive outs recorded (innings × 3)"),
        ("PO", "Putouts"),
        ("A", "Assists"),
        ("E", "Errors"),
        ("DP", "Double plays turned"),
        ("PB", "Passed balls (catchers)"),
        ("WP", "Wild pitches charged while fielding"),
        ("ZR", "Zone rating"),
        ("won_cy_young", "Indicator: won Cy Young Award"),
        ("won_mvp", "Indicator: won MVP Award"),
        ("won_gold_glove", "Indicator: won Gold Glove"),
        ("won_silver_slugger", "Indicator: won Silver Slugger"),
        ("all_star", "Indicator: made All-Star team"),
    ]

    pitchers_dict = [
        ("row_id", "Unique player-season identifier"),
        ("playerID", "Lahman player ID"),
        ("year", "Season year"),
        ("position", "Primary defensive position"),
        ("age", "Player age in the season"),
        ("avg_salary_year", "Average salary for the season"),
        ("free_agent_salary", "Salary agreed in free-agent deal for the season"),
        ("contract_length", "Length of the active contract (years)"),
        ("W", "Wins"),
        ("L", "Losses"),
        ("G", "Games pitched"),
        ("GS", "Games started"),
        ("CG", "Complete games"),
        ("SHO", "Shutouts"),
        ("SV", "Saves"),
        ("H", "Hits allowed"),
        ("ER", "Earned runs allowed"),
        ("HR", "Home runs allowed"),
        ("BB", "Walks allowed"),
        ("SO", "Strikeouts"),
        ("IBB", "Intentional walks issued"),
        ("WP", "Wild pitches"),
        ("HBP", "Hit batters"),
        ("BK", "Balks"),
        ("BFP", "Batters faced"),
        ("GF", "Games finished"),
        ("R", "Runs allowed"),
        ("SH", "Sacrifice hits allowed"),
        ("SF", "Sacrifice flies allowed"),
        ("GIDP", "Double plays induced"),
        ("ERA", "Earned run average"),
        ("BAOpp", "Opponent batting average"),
        ("InnOuts", "Outs recorded (innings × 3)"),
        ("PO", "Putouts while fielding"),
        ("A", "Assists while fielding"),
        ("E", "Errors while fielding"),
        ("DP", "Double plays fielded"),
        ("PB", "Passed balls (catchers)"),
        ("WP", "Wild pitches (duplicate fielding column)"),
        ("ZR", "Zone rating"),
        ("won_cy_young", "Indicator: won Cy Young Award"),
        ("won_mvp", "Indicator: won MVP Award"),
        ("won_gold_glove", "Indicator: won Gold Glove"),
        ("won_silver_slugger", "Indicator: won Silver Slugger"),
        ("all_star", "Indicator: made All-Star team"),
    ]

    dict_col1, dict_col2 = st.columns(2)
    with dict_col1:
        st.markdown("#### Batters dictionary")
        st.dataframe(pd.DataFrame(batters_dict, columns=["Column", "Description"]))
    with dict_col2:
        st.markdown("#### Pitchers dictionary")
        st.dataframe(pd.DataFrame(pitchers_dict, columns=["Column", "Description"]))

    st.markdown("## Final Datasets Overview")

    def load_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Batters (`final_batters_df.csv`)")
        if batters_path.exists():
            batters = load_csv(batters_path)
            st.write(f"Shape: {batters.shape[0]} rows × {batters.shape[1]} columns")
            st.dataframe(batters.sample(n=min(5, len(batters)), random_state=0))
        else:
            st.error(f"Missing dataset: {batters_path}")

    with col2:
        st.markdown("#### Pitchers (`final_pitchers_df.csv`)")
        if pitchers_path.exists():
            pitchers = load_csv(pitchers_path)
            st.write(f"Shape: {pitchers.shape[0]} rows × {pitchers.shape[1]} columns")
            st.dataframe(pitchers.sample(n=min(5, len(pitchers)), random_state=0))
        else:
            st.error(f"Missing dataset: {pitchers_path}")


if __name__ == "__main__":
    main()
