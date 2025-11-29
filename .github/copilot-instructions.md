## Project overview

- **Purpose:** Baseball salary analysis and predictive modeling for DS6021 course work.
- **Primary components:**
  - `dashboard/` — Streamlit app entry and pages (interactive UI). Run with `streamlit run Main.py` from the `dashboard/` folder.
  - `data/` — canonical CSV data used by notebooks and scripts (e.g., `Salaries.csv`, `Batting.csv`).
  - Top-level notebooks — analysis, EDA, and modeling live in `.ipynb` files (examples: `EDA.ipynb`, `Linear Regression Model (batters).ipynb`).

## Key facts for coding agents

- The UI is a Streamlit app under `dashboard/`. `dashboard/Main.py` is the launcher and `dashboard/pages/*` holds individual pages (e.g., `1_EDA.py`, `2_LinearRegression.py`). Keep dashboard files inside `dashboard/` only — README warns not to add unrelated files there.
- Data files live under `data/` and also at the repo root (e.g., `combined_free_agents_2014_2024.csv`, `final_batters_df.csv`). Use relative paths from the repository root when referencing CSVs in scripts/notebooks.
- Most reproducible analysis happens in notebooks. When converting notebook code to scripts, preserve the data-path conventions and double-check any notebook globals (plotting state, inline visuals) before refactoring.

## Run / debug conventions

- To run the dashboard (developer workflow):

  - Open a terminal in the repo and run:

    `cd dashboard`
    `streamlit run Main.py`

  - The pages use `st.set_page_config` and rely on Streamlit behavior (top-to-bottom re-runs). Avoid introducing long blocking operations at import time.

- Notebooks: open notebooks with your Jupyter environment. There is no centralized `requirements.txt`; typical dependencies are `streamlit`, `pandas`, `numpy`, `seaborn`, `matplotlib`.

## Patterns & project-specific conventions

- UI pages are small Streamlit scripts that use `st.*` calls directly (see `dashboard/pages/1_EDA.py`). Keep page logic focused on rendering — heavy processing should be moved to helper modules or pre-computed CSVs in `data/`.
- Visualization examples sometimes use sample datasets (e.g., `sns.load_dataset("tips")` in `2_LinearRegression.py`). Replace demo datasets with project CSVs when building production views.
- Notebooks are the canonical source of analysis; prefer minimal, well-documented notebook edits over large script-only rewrites.

## What to modify and what to avoid

- Modify: small Streamlit pages, helper scripts, and notebook cells when improving analysis or fixing bugs.
- Avoid: moving non-UI scripts into `dashboard/`, breaking relative data paths, or refactoring notebooks into scripts without a run/test pass.

## Helpful file references (examples)

- Launcher: `dashboard/Main.py`
- EDA page: `dashboard/pages/1_EDA.py`
- Regression demo: `dashboard/pages/2_LinearRegression.py`
- Data folder: `data/` (contains `Salaries.csv`, `Batting.csv`, etc.)
- Root CSVs and outputs: `combined_free_agents_2014_2024.csv`, `final_batters_df.csv`, `final_pitchers_df.csv`

## How to ask for clarification (when you are an assistant)

- If unsure about where to place code, ask: "Should this processing live in `dashboard/`, a notebook, or a new helper module?"
- If a notebook uses an absolute or unclear path, ask for the intended working directory and confirm whether inputs should be copied into `data/`.

---
If you want, I can also: add a `requirements.txt` generated from imports, extract common helper functions from notebooks into a `src/` module, or run the Streamlit app locally and report runtime errors. Which would you prefer next?
