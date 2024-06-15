import streamlit as st
import pandas as pd


@st.cache_data
def loadData():
    return pd.read_csv("./dataset/heart.csv")


@st.cache_data
def checkUnique(df):
    unique = []
    for i in df.columns:
        val = df[i].value_counts().count()
        unique.append(val)
    return pd.DataFrame(unique, index=df.columns, columns=["Total uniques values"])


class TableOfContent:
    def __init__(self):
        self.table = []

    def addAnchor(self, anchor):
        self.table.append(anchor)
        return anchor

    def genTableOfContent(self):
        pass
