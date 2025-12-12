# DS6021-Project
Final Project for DS6021 Predictive Modeling @UVA School of Data Science MSDS Program. <br>
Authors: Nathan Wan, Garret Knapp, Will Brannock, Joseph Kaminetz, Faizan Khan

## Overview
This project utilizes data from Lahman Baseball Database and MLB Stats API to generate both supervised and unsupervised machine learning models.

## Dashboard Info
Dashboard files are located in the `dashboard/` directory. To run the dashboard, navigate to that directory and run:
```
streamlit run Main.py
```
Please do not put any non dashboard files in that directory.

## Data Cleaning and EDA
Scripts for pulling and cleaning data and executing exploratory data analysis are located in this folder. 

## **Data**

This directory contains all datasets used in the project, divided into **raw** and **cleaned** subsets.

---

### **Raw Data**

Raw datasets come from two sources:

* the **Lahman Baseball Database** (via the *pylahman* package), and
* the **MLB Stats API** (via the *baseballr* package).

#### **1. Lahman**

The `lahman/` folder includes all raw CSV files pulled using *pylahman*. These are the unmodified datasets used throughout the project.

#### **2. MLB_stats_free_agents.csv**

This file contains MLB free-agent information from **2003–2015**, retrieved using the `mlb_people_free_agents()` function from *baseballr*.
For full details on how this data was collected, see the `free_agency_pull.r` script.

### **Cleaned Data**

This folder includes all datasets that have been cleaned, merged, or feature-engineered for use in the analysis and modeling stages.


## Modeling
Contains scripts for generating the following models for both Pitchers and Batters.

1. Linear Regression
2. Logistic Regression
3. KNN
4. KMeans
5. PCA
6. PCA Regression

## Final Presentation
Access the final presentation for this project [here](https://docs.google.com/presentation/d/1sAUolE4szf1zlqkt2Uhf214bI3dsnUf1DcD1E-f3liA/edit?usp=sharing)

# Deployed Dashboard
The dashboard is hosted at this [URL](https://ds6021-baseball-salary.streamlit.app/PCA_Regression)
