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

### **Exploratory Data Analysis** (reference EDA_FINAL.ipynb)
#### Batters: 
As you can see, the distribution of free agent salary is heavily focued on smaller contracts, with very few players receiving million dollar salaries. To do linear regression, we'll want to transform our response variable in order to see a normal distribution. Using a log transformation will achieve that.
The correlations between our features and response variables seem to make a lot of sense contextually to baseball. After seeing this, We see an ok to strong relationship with a lot of the positive hitting stats, especially runs and RBIs. We see a slight negative relationship with age which makes sense as if a younger player enters free agency, they will probably be values far more.
The VIFs values are extremely high, meaning to do any cogent analysis, we will have to run feature selection such as PCA or Lasso before almost every model. 

#### Pitchers: 
We see a similar distribution of salaries for pitchers as we did for batters, with a heavy focus on smaller contracts and very few large contracts. Again, we will want to log transform our response variable in order to do linear regression.
The correlation chart makes a lot of sense. The more strikeouts, innings, and wins a pitcher has, the more they make in free agency. We also see negative relationships with opponent's batting average and earned run average, both stats that pitchers are trying to avoid.
Similar to batters, we have far too high of VIFs. The same feature selection strategies must be implemented before modeling.

## **Model Conclusions**

### **Linear Regression**

#### Batters (reference Linear Regression Model (batters).ipynb)

Looking at our outputs, we see a test R^squared of 0.656, meaning that our model explains about 66% of the variation in each free agent's salary. This indicates that our model does a decent job at predicting free agent salary. Our model's transformed MSE is 0.341, meaning after retransforming, our models are off by a factor of about 1.6. This leads a bit to be desired, but overall isn't terrible at predicting free agent batter's salaries. Overall, our feature directions makes sense, as statistics that are thought of as good, like Home Runs, Hits, and Runs have a positive relationship, and statistics that are thought of as bad, like double plays and strikeouts see a negative relationship. Interestingly, RBIs, a stat that seems like it should have a positive relationship is negative. This may not mean a lot however, as it was deemed insignificant by the model.

#### Pitchers (reference Linear Regression Model (pitcher).ipynb)

Looking at our outputs, we see a test R^squared of 0.63, meaning that our model explains about 63% of the variation in each free agent's salary. This indicates that our model does a decent job at predicting free agent salary. Our model's transformed MSE is 0.403, meaning after retransforming, our models are off by a factor of about 1.9. This leads a bit to be desired, but overall isn't terrible at predicting pitcher's salaries. Overall, our feature directions makes sense, as statistics that are thought of as good, like Innings, Saves, and and strikeouts have a positive relationship, and statistics that are thought of as bad, like home runs and earned runs see a negative relationship. Looking at our features, we see that some of our very strong predictors appear to be volume based, features like innings pitched, saves, and wins. 

### **Logistic Regression**

#### Batters
FILL IN

#### Pitchers
FILL IN

### **KNN**

#### Batters (reference KNN- Batters - Lasso Selection.ipynb)

Our test r^squared of 0.68 indicates that our model does a pretty good job of predicting free agent salaries, explaining almost 70% of the variation. Our model MSE demonstrates that on average, our model misclassifies salaries by about \$2.8 million dollars. Considering the magnitude of some contracts, this seems reasonable. Looking at our predicted vs average graph, it seems that our model does a decent job of predicting salaries on the low end of the salary spectrum, but seems to overestimate salaries at the upper end. Our best K being 5 indicates that our data that is somewhat general, but still sensative to local changes.

#### Pitchers (reference KNN- Pitchers- Lasso Selection.ipynb)

Our test r^squared of 0.592 indicates that our model does an okay job of predicting free agent salaries, explaining almost 60% of the variation. Our model MSE demonstrates that on average, our model misclassifies salaries by about \$2.4 million dollars. Considering the magnitude of some contracts, this seems reasonable. Looking at our predicted vs average graph, it seems that our model does a decent job of predicting salaries throughout the salary ranges, but seems to overestimate salaries in the upper ranges. Our best K being 3 indicates that our data benefits from a model that is fairly local, giving more weight to the closest few points.

### **KMeans**

#### Batters (reference kmeans - batters.ipynb)

Based off our cluster analysis and comparisons of the characteristics of each group, we can say that cluster 0 appears to represent slightly older players on average who are more seasoned, focus on making contact rather than power plays (lower slugging percentage and strikeout rates). Meanwhile cluster 1 represents slightly younger players on average who are more "aggressive" players who prioritize power plays at cost of more strikeouts. Between these two groups, we see a statistically significant difference in salary and contract lengths, but investigating the standardized differences after accounting for outliers reveals that these differences are negligible for salary and small for contract length. So the older cluster 0 players make .2909 standard deviations less than cluster 1 players.

#### Pitchers (reference kmeans - pitchers.ipynb)

Based off the cluster analysis and cluster characteristic comparison visuals, we can say that cluster 0 appears to be older, less playtime players who play for a few power innings with higher strikeout rates, and allow less runs to happen. Meanwhile cluster 1 seems to represent younger, starting pitchers who allow more runs and have lower strikeout rates, probably due to the fact they are rookies and play more innings. Between these two groups, we see a statistically significant difference in salary and contract lengths, but investigating the standardized differences after accounting for outliers reveals that these differences are small for both comparisons, with older cluster 0 pitchers making about .2 standard deviations more on average.

### PCA Regression

#### Batters (reference models/PCA Regression- Batters.ipynb)

Using 5 principal components, the PCR model delivers a test R² of 0.571 (train R² 0.602) with a test RMSE of roughly $3.28M. That means
the compressed feature set still explains a little over half of the variance in free‑agent salaries, but typical errors are several million
dollars. The PCs largely capture overall offensive output and volume, so the model favors hitters with broad production profiles. Despite the
dimensionality reduction solving the multicollinearity issue, accuracy lags behind the plain linear model, suggesting important information was
lost when collapsing the feature space.

#### Pitchers (reference models/PCA Regression- Pitchers.ipynb)

With 4 components, the pitcher PCR reaches a test R² of 0.561 (train R² 0.591) and a test RMSE near $2.49M. The components primarily summarize
workload and run-prevention stats; the model rewards pitchers who log innings with strong run suppression but still misses finer salary
drivers. Like the batters’ model, PCR improves stability in the presence of correlated stats but underperforms the simpler regression in both
fit and error, indicating that reducing dimensionality trimmed away useful signal.
###
---
"""

st.markdown(introduction)
