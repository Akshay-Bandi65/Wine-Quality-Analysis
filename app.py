
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('winequality.csv')

st.title("Wine Quality Analysis Dashboard")

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Statistical Summary")
st.write(df.describe())

st.subheader("Wine Quality Distribution")
fig, ax = plt.subplots()
df['quality'].hist(ax=ax)
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)
