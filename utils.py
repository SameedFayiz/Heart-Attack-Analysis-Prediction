import streamlit as st
import pandas as pd


@st.cache_data
def loadData():
    return pd.read_csv("./dataset/heart.csv")
