import streamlit as st
import utils
from components.sidebar import viewSideBar
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import math

# Page configurations
st.set_page_config(layout="wide")
viewSideBar()
TOC = utils.TableOfContent()

# -----------------------------------------------------------
st.title("Heart Attack Prediction using Logistic Regression")


# Python implementation
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
X = df.drop("output", axis = 1)
y = df["output"]

# Training & Test data
st.subheader("Split training & test data",
             anchor=TOC.addSubAnchor("Data Cleaning & Preprocessing", "Split training & test data"))
st.code('''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
''')

# Feature Scaling
st.subheader("Feature scaling", anchor=TOC.addSubAnchor(
    "Data Cleaning & Preprocessing", "Feature scaling"))
st.code('''
from sklearn.preprocessing import StandardScaler
object = StandardScaler()
X_train = object.fit_transform(X_train)
X_test = object.transform(X_test)
print(f"Training set -> {X_train.shape}\\nTest set ->{X_test.shape}")
''')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
st.text(f"Training set -> {X_train.shape}\nTest set ->{X_test.shape}")

st.divider()

# Logistic Regression
st.header("Logistic Regression", anchor=TOC.addAnchor("Logistic Regression"))
st.subheader("Training the model", anchor=TOC.addSubAnchor(
    "Logistic Regression", "Training the model"))
scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(scaled_X_train, y_train)
st.code('''from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(scaled_X_train, y_train)''')
params = model.get_params()
st.write("Logistic Regression Model Parameters:", params)
coef=model.coef_
st.write("Logistic Regression Coefficients:",coef)

# Display model coefficients
st.code('''coes = pd.Series(model.coef_[0], index=X.columns)
coes = coes.sort_values(ascending = False)
st.write("Logistic Regression Coefficients:")
st.bar_chart(coes)''')
coes = pd.Series(model.coef_[0], index=X.columns)
coes = coes.sort_values(ascending = False)
st.write("Logistic Regression Coefficients:")
st.bar_chart(coes)

# Predict on the test set
st.code('''y_pred = model.predict(scaled_X_test)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax)
st.pyplot(fig)''')
y_pred = model.predict(scaled_X_test)

# Confusion matrix
st.code('''cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax)
st.pyplot(fig)''')
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax)
st.pyplot(fig)

# Classification report
st.write("Classification Report:")
st.code('''st.text(classification_report(y_test, y_pred))''')
st.text(classification_report(y_test, y_pred))

# Precision-recall curve
st.code('''y_true = [12, 3]
y_scores = [2, 14]
precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=12)
fig, ax = plt.subplots()
ax.plot(recall, precision, marker='o', label='Precision-Recall Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(loc='best')
ax.grid(True)
st.pyplot(fig)''')
y_true = [12, 3]
y_scores = [2, 14]
precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=12)
fig, ax = plt.subplots()
ax.plot(recall, precision, marker='o', label='Precision-Recall Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(loc='best')
ax.grid(True)
st.pyplot(fig)

st.code('''accuracy = accuracy_score(y_test, y_pred)''')
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)

#python
st.header("using python")
st.code('''data = df.to_dict(orient='records')
X = []
y = []
for row in data:
    features = [value for key, value in row.items() if key != 'output']
    X.append(features)
    y.append(row['output'])
def train_test_split(X, y, test_size=0.1, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    data = list(zip(X, y))
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]
    X_train = [x for x, _ in train_data]
    y_train = [y for _, y in train_data]
    X_test = [x for x, _ in test_data]
    y_test = [y for _, y in test_data]
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

def fit_scaler(X):
    mean = [sum(col) / len(col) for col in zip(*X)]
    std = [(sum((x - m) ** 2 for x in col) / len(col)) ** 0.5 for col, m in zip(zip(*X), mean)]
    return mean, std
def transform_scaler(X, mean, std):
    scaled_X = [[(x - m) / s for x, m, s in zip(row, mean, std)] for row in X]
    return scaled_X
mean, std = fit_scaler(X_train)
scaled_X_train = transform_scaler(X_train, mean, std)
scaled_X_test = transform_scaler(X_test, mean, std)

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
weights, bias = gradient_descent(scaled_X_train, y_train, learning_rate, num_iterations)

y_pred = predict(scaled_X_test, weights, bias)

''')


data = df.to_dict(orient='records')
X = []
y = []

for row in data:
    features = [value for key, value in row.items() if key != 'output']
    X.append(features)
    y.append(row['output'])

def train_test_split(X, y, test_size=0.1, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    data = list(zip(X, y))
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]
    X_train = [x for x, _ in train_data]
    y_train = [y for _, y in train_data]
    X_test = [x for x, _ in test_data]
    y_test = [y for _, y in test_data]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

def fit_scaler(X):
    mean = [sum(col) / len(col) for col in zip(*X)]
    std = [(sum((x - m) ** 2 for x in col) / len(col)) ** 0.5 for col, m in zip(zip(*X), mean)]
    return mean, std
def transform_scaler(X, mean, std):
    scaled_X = [[(x - m) / s for x, m, s in zip(row, mean, std)] for row in X]
    return scaled_X
mean, std = fit_scaler(X_train)
scaled_X_train = transform_scaler(X_train, mean, std)
scaled_X_test = transform_scaler(X_test, mean, std)

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
weights, bias = gradient_descent(scaled_X_train, y_train, learning_rate, num_iterations)

y_pred = predict(scaled_X_test, weights, bias)

st.code('''cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:", cm)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax)
st.pyplot(fig)''')
cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:", cm)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax)
st.pyplot(fig)


# Classification report
st.write("Classification Report:")
st.code('''classification_report(y_test, y_pred)''')
st.text(classification_report(y_test, y_pred))

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

precision, recall, threshold = precision_recall_curve(y_true, y_scores, pos_label = 12)

plt.plot(precision, recall, marker = "o", label = "Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precsion")
plt.legend(loc = "best")
plt.title("Precision Recall Curve")
plt.grid(True)
plt.show()

st.code('''accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)
''')

accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)

st.divider()
TOC.genTableOfContent()