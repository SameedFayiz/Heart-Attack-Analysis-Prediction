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
st.code('''import pandas as pd
data = pd.read_csv('./dataset/heart.csv')
data.head()''')

df = utils.loadData()
sample = df.head()
st.dataframe(sample, use_container_width=True)

# Dataset description
st.subheader("Dataset description")
st.write('''The following variables are included in the dataset:
- **age**: Age of the patient
- **sex**: Sex of the patient ( 0: Female, 1: Male )
- **cp**: Chest pain type ( 0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic )
- **trtbps**: Resting blood pressure ( in mm Hg)
- **chol**: Cholesterol level fetched via BMI sensor ( in mg/dl )
- **fbs**: Fasting blood sugar ( 0: <= 120 mg/dl, 1: > 120 mg/dl )
- **restecg**: Resting electrocardiographic results ( 0: Normal, 1: ST-T wave abnormality, 2: Probable or definite left ventricular hypertrophy by Estes' criteria )
- **thalachh**: Max heart rate achieved
- **exng**: Exercise induced angina ( 0: Yes, 1: No )
- **oldpeak**: ST depression induced by exercise relative to rest
- **slp**: The slope of the peak exercise ST segment ( 0: Upsloping, 1: Flat, 2: Downsloping )
- **caa**: Number of major vessels ( 0-3 )
- **thall**: Thalassemia (0: Null, 1: Fixed defect, 2: Normal, 3: Reversible defect)
- **output**: Diagnosis of heart disease ( 0: No, 1: Yes )''')

st.text(
    f"No. of Columns {len(df.columns)}\nNo. of Rows {df.shape[0]}")
st.write("Check null values in dataset:")
st.code("data.isnull().sum()")
st.text(f"{df.isnull().sum()}")

# Check unqiue values
st.subheader("Examining for Unique values")
st.code('''unique=[]
for i in df.columns:
    val=df[i].value_counts().count()
    unique.append(val)
df_=pd.Dataframe(unique, index=df.columns, columns['Total uniques values'])
df_.T''')

st.dataframe(utils.checkUnique(df).T, use_container_width=True)

st.write('''Upon inspection of the above dataframe, it can easily be inferred that:
- "sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall", "output" are **Categorical variables**
- “age”, “trtbps”, “chol”, “thalachh” and “oldpeak” are **Numeric variables**\n
Now, separating both categories for further analysis''')
st.code('''
numeric_vars = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
categoric_vars = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall", "output"]''')
numeric_vars = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
categoric_vars = ["sex", "cp", "fbs", "restecg",
                  "exng", "slp", "caa", "thall", "output"]

# Analysis of categorical variables
st.subheader("Categorical variables analysis")

st.code("categoric_vars")
st.text(categoric_vars)
st.code('''
cat_graph_names = ["Gender", "Chest Pain Type", "Fasting Blood sugar", "Resting Electrocardiographic Results",
                   "Exercise Induced Angina", "The Slope of ST Segment", "No. of Major Vessels", "Thal", "Target"]''')
cat_graph_names = ["Gender", "Chest Pain Type", "Fasting Blood sugar", "Resting Electrocardiographic Results",
                   "Exercise Induced Angina", "The Slope of ST Segment", "No. of Major Vessels", "Thal", "Target"]
