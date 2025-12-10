import streamlit as st

st.set_page_config(
    page_title="Baseball Salary Analysis",
    page_icon="⚾",
)

st.title("DS 6021 Final Project – Baseball Salary Analysis")
st.write("Team Members: Nathan Wan, Garret Knapp, Will Brannock, Joseph Kaminetz, and Faizan Khan")
st.sidebar.success("Select a page above.")

introduction = """


# Project Overview

## Problem Statement

One of the most chaotic and anticipated parts of every MLB season is the offseason, when teams compete for talent and fans speculate about the contracts players will receive. Free-agent negotiations are shaped by a complex mix of performance metrics, player reputation, market conditions, age, and positional value. Despite their importance, predicting contract outcomes remains challenging due to the number of factors involved and the uncertainty inherent in player valuation.

Given this complexity, this project focuses specifically on understanding the relationship between player performance and the resulting salary and contract length. To investigate these dynamics, we developed both supervised and unsupervised learning models designed to answer the following questions:

1. **Given player awards, performance statistics, league-average salary, age, and other relevant factors, can we predict a player’s annual average salary (AAV)?**
2. **Can we also predict whether a player will sign a multi-year contract?**

The following models were developed to answer these questions:
* Linear Regression
* Binary Logistic
* PCA
* PCA Regression
* KNN
* KMeans

For more information please visit the respective tabs for each of these models.

---

## Approach

### **Data and Cleaning**
To answer these questions, we require detailed information on free agents, including their performance statistics, awards, salaries, and contract lengths following free agency. This necessitates high-fidelity data covering player performance, free-agent classes, and the contracts signed during each offseason. In this project, the **Lahman Baseball Dataset** and the **MLB Stats API** are combined to create a cleaned dataset of free agents spanning **2003–2015**. For more information on these data sources, please refer to the **Datasets** section under the **Data and Cleaning** tab.

Before modeling, the data must be carefully cleaned and structured. Below are the key decisions regarding how player statistics and awards are handled:

* **Performance statistics** are averaged or accumulated depending on the nature of the metric, using the three seasons prior to free agency.
    * *Example:* For a free agent in the 2010 offseason, statistics from the 2008–2010 seasons are aggregated accordingly.
* **Awards** are converted into categorical variables indicating whether a player received specific awards within the three seasons leading up to free agency.
* **AAV** (Average Annual Value) is calculated by averaging the salaries across the length of the contract signed following free agency.
* **Contract Type** the focus of this analysis is MLB contracts so Minor Leqgue contracts will not be analyzed.

Additionally, due to significant differences in statistics between pitchers and position players, we separate batters and pitchers to provide better fidelity for both groups. As a result, all models developed to answer the research questions above include a **batter-specific version** and a **pitcher-specific version**. These decisions produce two cleaned datasets—one for pitchers and one for batters—where every free agent from 2003–2015 is matched with performance statistics, awards, and the resulting contract AAV along with other attributes.

For further details on the data cleaning process, please refer to the **Data Cleaning** section under the **Data and Cleaning** tab. 

### **Exploratory Data Analysis**

## Conclusions

TODO

---
"""

st.markdown(introduction)
