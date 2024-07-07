import streamlit as st
import utils
import pandas as pd
import numpy as np
from components.sidebar import viewSideBar
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve, accuracy_score
from models.logRegress import *

# Page configurations
st.set_page_config(layout="wide")
viewSideBar()

st.link_button(
    "Open in Google Colab", "https://colab.research.google.com/drive/18Nm1NUA3qhsZo6Ay1aNoV0rdl3IoCqpc?usp=sharing")

TOC = utils.TableOfContent()

# -----------------------------------------------------------
st.title("Heart Attack Prediction using Logistic Regression")

st.header("Data Cleaning & Preprocessing",
          anchor=TOC.addAnchor("Data Cleaning & Preprocessing"))

# Load and read the data
st.subheader("Load & view data", anchor=TOC.addSubAnchor(
    "Data Cleaning & Preprocessing", "Load & view data"))
st.code('''import pandas as pd
data = pd.read_csv('./dataset/heart.csv')
print(f"No. of Columns {len(df.columns)}\\nNo. of Rows {df.shape[0]})
data.head()''')

df = utils.loadData()
sample = df.head()
st.dataframe(sample, use_container_width=True)
st.text(f"No. of Columns {len(df.columns)}\nNo. of Rows {df.shape[0]}")

# Check for null values
st.subheader("Check for null values",
             anchor=TOC.addSubAnchor("Data Cleaning & Preprocessing", "Check for null values"))
st.code("data.isnull().sum()")
st.text(f"{df.isnull().sum()}")

# Check for duplicates
st.subheader("Check for duplicates",
             anchor=TOC.addSubAnchor("Data Cleaning & Preprocessing", "Check for duplicates"))
st.code("df[df.duplicated()")
st.write(df[df.duplicated()])
st.write("Drop duplicate row")
st.code('''df.drop_duplicates()
print(f"No. of Columns {len(df.columns)}\\nNo. of Rows {df.shape[0]})''')

df = df.drop_duplicates()
st.text(f"No. of Columns {len(df.columns)}\nNo. of Rows {df.shape[0]}")

# Separate features & target variable
st.subheader("Features & target variables",
             anchor=TOC.addSubAnchor("Data Cleaning & Preprocessing", "Features & target variables"))
st.code('''import numpy as np
X = df.drop("output", axis = 1) # Features
y = df["output"] # Target''')
X = df.drop("output", axis=1)
y = df["output"]

# Training & Test data
st.subheader("Split training & test data",
             anchor=TOC.addSubAnchor("Data Cleaning & Preprocessing", "Split training & test data"))
st.code('''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
''')

# Feature Scaling
st.subheader("Feature scaling", anchor=TOC.addSubAnchor(
    "Data Cleaning & Preprocessing", "Feature scaling"))
st.code('''
from sklearn.preprocessing import StandardScaler
object = StandardScaler()
X_train_scaled = object.fit_transform(X_train)
X_test_scaled = object.transform(X_test)
print(f"Training set -> {X_train.shape}\\nTest set ->{X_test.shape}")
''')

X_train, X_test, y_train, y_test, object = utils.preProcessAndSplit(df)
st.text(f"Training set -> {X_train.shape}\nTest set ->{X_test.shape}")

st.divider()

# Logistic Regression
st.header("Logistic Regression", anchor=TOC.addAnchor("Logistic Regression"))
st.subheader("Training the model", anchor=TOC.addSubAnchor(
    "Logistic Regression", "Training the model"))


model = LogisticRegression()
model.fit(X_train, y_train)
st.code('''from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)''')
params = model.get_params()
st.write("Logistic Regression Model Parameters:", params)
coef = model.coef_
st.write("Logistic Regression Coefficients:")
st.dataframe(coef, use_container_width=True)

# Display model coefficients
st.code('''coes = pd.Series(model.coef_[0], index=X.columns)
coes = coes.sort_values(ascending = False)
st.write("Logistic Regression Coefficients:")
st.bar_chart(coes)''')
coes = pd.Series(model.coef_[0], index=X.columns)
coes = coes.sort_values(ascending=False)
st.write("Logistic Regression Coefficients:")
st.bar_chart(coes)

# Predict on the test set
st.code("y_pred = model.predict(X_test_scaled)")
y_pred = model.predict(X_test)
st.write(y_pred.reshape(1, -1))

st.subheader("Predict Yourself ",
             anchor=TOC.addSubAnchor("Logistic Regression", "Predict Yourself "))


def predictor(x):
    x = object.transform(x)
    st.dataframe(pd.DataFrame(x, columns=X.columns), use_container_width=True)

    y = model.predict(x)[0]
    if y > 0.5:
        st.error(
            f"Patient has high chances of heart attack")
    else:
        st.success(f"Patient has very low chances of heart attack")


utils.fragment_function(predictor, key=2)


# Evaluation
st.subheader("Model evaluation",
             anchor=TOC.addSubAnchor("Logistic Regression", "Model evaluation"))

st.write("**Accuracy:**")
st.code('''
from sklearn.metrics import accuracy_score
print(f'Accuracy: { accuracy_score(y_test,y_pred) :.3}\\n')
''')
st.write("Accuracy: ", float(f"{accuracy_score(y_test,y_pred) :.3}"))

# Confusion matrix
st.write("**Confusion matrix:**")
st.code('''
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax)''')

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax)
st.pyplot(fig)

# Classification report
st.write("**Classification Report:**")
report = classification_report(
    y_test, y_pred, output_dict=True, target_names=["Class 0", "Class 1"])
report.update({"accuracy": {"precision": None, "recall": None,
              "f1-score": report["accuracy"], "support": report['macro avg']['support']}})
st.dataframe(pd.DataFrame(report).T, use_container_width=True)

# Precision-recall curve
st.write("**Precision-recall curve:**")
st.code('''y_true = [12, 2]
y_scores = [1, 16]
precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=12)
fig, ax = plt.subplots()
ax.plot(recall, precision, marker='o', label='Precision-Recall Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(loc='best')
ax.grid(True)''')

y_true = [12, 2]
y_scores = [1, 16]
precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=12)
fig, ax = plt.subplots()
ax.plot(recall, precision, marker='o', label='Precision-Recall Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(loc='best')
ax.grid(True)
st.pyplot(fig)


# python
st.header("Logistic Regression (Without python package)",
          anchor=TOC.addAnchor("Logistic Regression (Without python package)"))
st.subheader("Defining & training the model", anchor=TOC.addSubAnchor(
    "Logistic Regression (Without python package)", "Defining & training the model"))
st.code('''
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict_proba(X, weights, bias):
    return [sigmoid(sum(w * x for w, x in zip(weights, row)) + bias) for row in X]

def predict(X, weights, bias):
    return [1 if p >= 0.5 else 0 for p in predict_proba(X, weights, bias)]

def gradient_descent(X, y, learning_rate, num_iterations):
    n_samples, n_features = len(X), len(X[0])
    weights = [0] * n_features
    bias = 0

    for _ in range(num_iterations):
        y_pred = predict_proba(X, weights, bias)
        dw = [0] * n_features
        db = 0

        for i in range(n_samples):
            error = y_pred[i] - y[i]
            for j in range(n_features):
                dw[j] += error * X[i][j]
            db += error

        weights = [w - learning_rate * dw[j] / n_samples for j, w in enumerate(weights)]
        bias -= learning_rate * db / n_samples

    return weights, bias

learning_rate = 0.01
num_iterations = 1000
weights, bias = gradient_descent(X_train_scaled, y_train, learning_rate, num_iterations)

y_pred = predict(X_test_scaled, weights, bias)

''')


learning_rate = 0.01
num_iterations = 1000
weights, bias = gradient_descent(
    X_train, y_train, learning_rate, num_iterations)

y_pred = predict(X_test, weights, bias)

st.subheader("Predict Yourself",
             anchor=TOC.addSubAnchor("Logistic Regression (Without python package)", "Predict Yourself"))


def predictor(x):
    x = object.transform(x)
    st.dataframe(pd.DataFrame(x, columns=X.columns), use_container_width=True)

    y = predict(x, weights, bias)[0]
    if y > 0.5:
        st.error(
            f"Patient has high chances of heart attack")
    else:
        st.success(f"Patient has very low chances of heart attack")


utils.fragment_function(predictor, key=3)

# Evaluation
st.subheader("Model Evaluation",
             anchor=TOC.addSubAnchor("Logistic Regression (Without python package)", "Model Evaluation"))

st.write("**Accuracy:**")
st.code('''
from sklearn.metrics import accuracy_score
print(f'Accuracy: { accuracy_score(y_test,y_pred) :.3}\\n')
''')
st.write("Accuracy: ", float(f"{accuracy_score(y_test,y_pred) :.3}"))

# Confusion matrix
st.write("**Confusion matrix:**")
st.code('''cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax)''')
cm = confusion_matrix(y_test, y_pred)
st.write(cm)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax)
st.pyplot(fig)


# Classification report
st.write("**Classification Report:**")
st.code('''classification_report(y_test, y_pred)''')
report = classification_report(
    y_test, y_pred, output_dict=True, target_names=["Class 0", "Class 1"])
report.update({"accuracy": {"precision": None, "recall": None,
              "f1-score": report["accuracy"], "support": report['macro avg']['support']}})
st.dataframe(pd.DataFrame(report).T, use_container_width=True)


# Precision-recall curve
st.write("**Precision-recall curve:**")
st.code('''y_true = [12, 5]
y_scores = [0, 14]

precision, recall, threshold = precision_recall_curve(y_true, y_scores, pos_label = 12)

plt.plot(precision, recall, marker = "o", label = "Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precsion")
plt.legend(loc = "best")
plt.title("Precision Recall Curve")
plt.grid(True)
plt.show()''')
y_true = [12, 5]
y_scores = [0, 14]

precision, recall, threshold = precision_recall_curve(
    y_true, y_scores, pos_label=12)

fig, ax = plt.subplots()
ax.plot(recall, precision, marker='o', label='Precision-Recall Curve')
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(loc="best")
ax.set_title("Precision-Recall Curve")
ax.grid(True)
st.pyplot(fig)

st.divider()
TOC.genTableOfContent()
