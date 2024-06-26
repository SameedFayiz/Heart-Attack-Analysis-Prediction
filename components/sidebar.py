import streamlit as st

# Sidebar configuration
def viewSideBar():
    st.sidebar.page_link("index.py", label="Home")
    st.sidebar.page_link("pages/eda.py", label="Exploratory Data Analysis")
    st.sidebar.page_link("pages/neuralNetwork.py",
                         label="Neural Network model")
    st.sidebar.page_link("pages/logisticregression.py",
                         label="Logistic Regression model")

    st.sidebar.divider()

    st.sidebar.link_button(
        "Go to GitHub Repo", "https://github.com/SameedFayiz/Heart-Attack-Analysis-Prediction", type="primary")
