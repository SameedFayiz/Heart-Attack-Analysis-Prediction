import streamlit as st
import pandas as pd
import utils
from components.sidebar import viewSideBar
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
main=st.container()
with main:
    st.subheader("Data Visualization")
    st.write("Select a variable to visualize:")
    variable = st.selectbox("Variable", numeric_vars)
    st.write(f"You selected: {variable}")
    st.write("Visualize the selected variable:")
    st.bar_chart(df[[variable]])
    unique_number = []
    for i in df.columns:
        x = df[i].value_counts().count()
        unique_number.append(x)
    st.write(pd.DataFrame(unique_number, index=df.columns, columns=["Total Unique Values"]))
    st.write("Select a variable to visualize:")
    variable_2 = st.selectbox("Variable", categoric_vars)
    st.write(f"You selected: {variable_2}")
    st.write("Visualize the selected variable:")
    pie_data = df[variable_2].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal') 
    st.pyplot(fig)
    st.subheader("Value Counts")
    st.write(df[variable_2].value_counts())
tbh=st.container()
with tbh:
    st.subheader("Descriptive Statistics")
    st.write("Descriptive statistics of numerical variables:")
    st.write(df[numeric_vars].describe())
    null_number = []
    x = df[i].isnull().sum()
    for i in df.columns:
        null_number.append(x)
    st.write("Check correlation between variables:")
    st.write(pd.DataFrame(null_number, index=df.columns, columns=["Missing Values"]))
    figure = plt.figure(figsize = (10,10))
    st.write(df[categoric_vars].corr())
    sns.heatmap(df.corr(),annot=True)
    st.pyplot(figure)
    st.write("The correlation heatmap shows the most related, important, and effective attributes on the target attribute. It is seen that thalachh (the maximum heart rate) and the age is negatively correlated as it has been shown above. (-0.4) Also, cp (chest pain type) is positively highly correlated with the output directly. (0.43) Thallachh, cp, and sex are selected.")
    st.write("now investigating their effects")
    male = df[df['sex'] == 1]
    female = df[df['sex'] == 0]
    male_counts = male['output'].value_counts()
    female_counts = female['output'].value_counts()
    colors = ['blue', 'red']  
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].pie(male_counts, labels=['No Heart Attack', 'Heart Attack'], colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
    axes[0].set_title('Male')
    axes[1].pie(female_counts, labels=['No Heart Attack', 'Heart Attack'], colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
    axes[1].set_title('Female')
    fig.suptitle('The Effect of Sex on Risk of Heart Attack (Output)', fontsize=18)
    st.pyplot(fig)
    st.write('The pie chart shows that more males (44.7%) tend to have heart attacks compared to females (25%)')
    st.write("Now let's investigate the effect of chest pain type (cp) on the risk of heart attack (output).")
    fig = px.histogram(df, x='cp', color='output',barmode='group',color_discrete_sequence=px.colors.qualitative.Set2,labels={'cp': 'Chest Pain Types', 'output': 'Output'})
    fig.update_xaxes(showgrid=True, tickvals=[0, 1, 2, 3],gridcolor='black',
                 ticktext=['Typical Angina (0)', 'Atypical Angina (1)', 'Non-Anginal Pain (2)', 'Asymptomatic (3)'])
    fig.update_yaxes(showgrid=True,gridcolor='black')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig)
    st.write('People with chest pain type 2 (Non-Anginal pain) found to have more heart attacks.')
    st.write('now thall')
    fig = px.histogram(df, x='thall', color='output', title='The Effect of Thall and Risk of Heart Attack (Output)',
                   labels={'thall': 'Thal Type', 'output': 'Output'}, barmode='group',
                   color_discrete_sequence=['green', 'blue'])
    fig.update_layout(
        bargap=0.1
    )
    fig.update_xaxes(showgrid=True, gridcolor='black', tickvals=[0, 1, 2, 3], 
                    ticktext=['None (Normal) (0)', 'Fixed Defect (1)', 'Reversible Defect (2)', 'Thalassemia (3)'])
    fig.update_yaxes(showgrid=True, gridcolor='black')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig)
    st.write('People with Thall type 2 Reversible Defect have more chances of heart attacks.')
    st.subheader("Conclusion")
    st.write("From the above analysis, it can be concluded that:")
    st.write("- Age, chest pain type, and thall type are the most important factors affecting the risk of heart attack.")
    st.write('thalachh')
    fig = px.box(
        df,
        x='output',
        y='thalachh',
        labels={'output': 'Heart Disease Output', 'thalachh': 'Maximum Heart Rate Achieved'},
        title='Distribution of Maximum Heart Rate Achieved by Heart Disease Output',
        color='output', 
        color_discrete_sequence=px.colors.qualitative.Set2  
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis_title_font=dict(size=16, family='Arial', color='black'),
        yaxis_title_font=dict(size=16, family='Arial', color='black'),
        xaxis=dict(
            tickvals=[0, 1],
            ticktext=['No Heart Disease (0)', 'Heart Disease (1)']
        ),
        margin=dict(l=50, r=50, t=70, b=50)  # Adjust margins for better spacing
    )
    st.plotly_chart(fig)
    st.write('The maximum heart rate achieved by patients with heart disease is higher compared to patients without heart disease.')
    
