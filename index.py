import streamlit as st
import pandas as pd
from components.sidebar import viewSideBar

# Page configurations
st.set_page_config(layout="wide")
viewSideBar()

# ----------------------------------------------
st.title("Heart attack Analysis & Prediction")
st.write("This repository contains a project focused on analyzing and predicting heart attacks using machine learning techniques. The application is built with Streamlit, providing an interactive interface for users to explore and predict heart.")
