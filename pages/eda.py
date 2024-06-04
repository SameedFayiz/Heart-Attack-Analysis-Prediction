import streamlit as st
import pandas as pd
import utils
from components.sidebar import viewSideBar

# Page configurations
st.set_page_config(layout="wide")
viewSideBar()

# -----------------------------------------------------------
st.header("Exploratory Data Analysis of Heart Attack Data")
# Load and read the data
st.subheader("Load & view data")
st.code("import pandas as pd\ndata = pd.read_csv('./dataset/heart.csv')\ndata.head()")

csvData = utils.loadData()
sample = csvData.head()
st.dataframe(sample, use_container_width=True)

# Dataset description
st.subheader("Dataset description")
st.write('''The following variables are included in the dataset:
- Age: Age of the patient
- Sex: Sex of the patient
- Exang: Exercise induced angina ( 1: Yes, 2: No )
- Ca: Number of major vessels(0-3)
- Cp: Chest pain type ( 1: Typical angina, 2: Atypical angina, 3: Non-anginal pain, 4: Asymptomatic )
- Trtbps: Resting blood pressure ( in mm Hg)
- Chol: Cholesterol level in mg/dl(fetched via BMI sensor)
- Restecg: Resting electrocardiographic results ( 0: Normal, 1: ST-T wave abnormality, 2: Probable or definite left ventricular hypertrophy by Estes' criteria )
- Thalachh: Max heart rate achieved ( 0: Less chance of heart attack, 1: More chance of heart attack )
''')

st.text(
    f"No. of Columns {len(csvData.columns)}\nNo. of Rows {csvData.size}")
st.write("Check null values in dataset:")
st.code("data.isnull().sum()")
st.text(f"{csvData.isnull().sum()}")
