import streamlit as st

# Sidebar configuration
def viewSideBar():
    st.sidebar.page_link("index.py", label="Home")
    st.sidebar.page_link("pages/eda.py", label="Exploratory Data Analysis")
    # st.sidebar.page_link("", label="")
    # st.sidebar.page_link("", label="")

    st.sidebar.divider()

    st.sidebar.link_button(
        "Go to GitHub Repo", "https://github.com/SameedFayiz/Heart-Attack-Analysis-Prediction")
