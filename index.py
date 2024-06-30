import streamlit as st
import pandas as pd
from components.sidebar import viewSideBar

# Page configurations
st.set_page_config(layout="wide")
viewSideBar()

# ----------------------------------------------
st.title("Heart attack Analysis & Prediction")
st.write("Welcome to the Heart Attack Analysis & Prediction application! This project aims to provide a comprehensive tool for analyzing and predicting heart attacks using state-of-the-art machine learning techniques. Built with Streamlit, our interactive interface allows users to explore data insights and make predictions with ease.")

st.header("Overview")
st.write("Cardiovascular diseases are the leading cause of death globally, making heart attack prediction a critical area of research. Leveraging machine learning, we can gain deeper insights into the factors contributing to heart attacks and develop models that can predict their occurrence with high accuracy.")

st.subheader("Dataset source")
st.page_link(
    "https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset", label="https: // www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset")

st.header("Features")
st.write('''
### 1. Exploratory Data Analysis (EDA)
Our application begins with an in-depth exploratory data analysis to understand the underlying patterns and correlations within the dataset. Key features include:

- Visualization of important variables
- Correlation heatmaps
- Distribution plots for different health metrics
- Statistical summaries
### 2. Predictive Modeling
We have implemented two powerful predictive models to estimate the risk of heart attacks:

#### Logistic Regression
**From Scratch**: A custom implementation of logistic regression to illustrate the underlying mechanics.\n
**Scikit-learn**: Utilizes the widely-used Scikit-learn library for a robust and efficient implementation.
#### Neural Network
**From Scratch**: A foundational neural network built from the ground up to demonstrate deep learning principles.\n
**TensorFlow**: A sophisticated neural network model using TensorFlow, one of the leading deep learning libraries.
### 3. User-friendly Interface
Our Streamlit-powered interface ensures that users, regardless of their technical background, can easily interact with the application. Features include:

- Interactive data visualization tools
- Easy-to-use prediction interfaces
- Customizable input parameters for personalized risk assessment
### 4. Model Evaluation
Comprehensive evaluation metrics to assess the performance of our models, including:

- Accuracy, precision, recall, and F1 score
- Confusion matrices
- Classification report''')
