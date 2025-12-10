from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Data Cleaning", page_icon="🧹")


@st.cache_data
def load_markdown_cells(notebook_path: Path, mtime: float) -> list[str]:
    notebook_json = json.loads(notebook_path.read_text())
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

    if not notebook_path.exists():
        st.error(f"Notebook not found at {notebook_path}")
        return

    st.sidebar.header("Data Cleaning")
    st.title("Data Cleaning")

    sections = load_markdown_cells(notebook_path, notebook_path.stat().st_mtime)
    render_markdown_sections(sections)


if __name__ == "__main__":
    main()
