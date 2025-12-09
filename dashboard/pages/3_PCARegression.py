import streamlit as st
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Linear Regression", page_icon="📈")

st.markdown("# Linear Regression")
st.sidebar.header("Linear Regression")
st.write(
    """This shows an example linear regression plot using seaborn"""
)

# create regression chart using seaborn and display it

# load example dataset
df = sns.load_dataset("tips")
# create regression plot
plt.figure(figsize=(10,6))
sns.regplot(x="total_bill", y="tip", data=df)
plt.title("Linear Regression of Tip vs Total Bill")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
st.pyplot(plt)
