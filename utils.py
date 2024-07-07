import pandas as pd
import numpy as np
import urllib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.neuralNet import NeuralNetwork, FullyConnectedLayer, ActivationLayer
from models.activations import sigmoid, sigmoid_prime, Relu, Relu_prime, BinaryCrossEntropy, BinaryCrossEntropyPrime


def gradient_color(val):
    r = 255
    g = 255 - val*2
    b = 0
    return f'background-color: rgb({r},{g},{b})'


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
        self.table = {}

    def addAnchor(self, anchor):
        self.table[anchor] = []
        return anchor

    def addSubAnchor(self, head, anchor):
        self.table[head].append(anchor)
        return anchor

    def genTableOfContent(self):
        mk = ""
        st.sidebar.subheader("Table of Contents")
        for i in self.table.items():
            url = urllib.parse.quote(i[0])
            mk += f"- [{i[0]}](#{url})\n"
            for j in i[1]:
                url = urllib.parse.quote(j)
                mk += f"    - [{j}](#{url})\n"

        st.sidebar.markdown(mk)


@st.cache_data
def preProcessAndSplit(data):
    X = np.array(data.drop('output', axis=1))  # Features
    y = np.array(data['output'])  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    object = StandardScaler()
    X_train = object.fit_transform(X_train)
    X_test = object.transform(X_test)

    return X_train, X_test, y_train, y_test, object


def trainNNModel(X_train, y_train, eleRef):
    def writeEpochs(ep, eps, error):
        eleRef.write(f"Epoch {ep+1}/{eps}, Error = {error :.3f}")

    net = NeuralNetwork()
    net.add(FullyConnectedLayer(13, 13))
    net.add(ActivationLayer(Relu, Relu_prime))
    net.add(FullyConnectedLayer(13, 26))
    net.add(ActivationLayer(Relu, Relu_prime))
    net.add(FullyConnectedLayer(26, 1))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))
    net.use(BinaryCrossEntropy, BinaryCrossEntropyPrime)

    return net, net.fit(X_train, y_train, epochs=1000, learning_rate=0.002, printCallback=writeEpochs)


@st.experimental_fragment
def fragment_function(callBack, key):
    cat_options = {"Sex": {"Female": 0, "Male": 1},
                   "Chest pain": {"Typical angina": 0,  "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3},
                   "Fasting blood sugar": {"<= 120 mg/dl": 0, "> 120 mg/dl": 1},
                   "Resting electrocardiographic results": {"Normal": 0, "ST-T wave abnormality": 1, "Probable left ventricular hypertrophy": 2},
                   "Exercise induced angina": {"Yes": 0, "No": 1},
                   "Slope (peak exercise ST segment)": {"Upsloping": 0, "Flat": 1, "Downsloping": 2},
                   "Major vessels": {0: 0, 1: 1, 2: 2, 3: 3},
                   "Thalassemia": {"Null": 0,  "Fixed defect": 1, "Normal": 2, "Reversible defect": 3}}

    num_ranges = {"Age": [20, 100],
                  "Resting blood pressure": [94, 200],
                  "Cholesterol level": [126, 564],
                  "Max heart rate achieved": [71, 202],
                  "ST depression induced by exercise relative to rest": [0.0, 6.2]}

    test_vals = {"Age": 54,
                 "Sex": 1,
                 "Chest pain": 0,
                 "Resting blood pressure": 131,
                 "Cholesterol level": 246,
                 "Fasting blood sugar": 0,
                 "Resting electrocardiographic results": 0,
                 "Max heart rate achieved": 149,
                 "Exercise induced angina": 0,
                 "ST depression induced by exercise relative to rest": 1,
                 "Slope (peak exercise ST segment)": 0,
                 "Major vessels": 0,
                 "Thalassemia": 0
                 }

    for i in num_ranges.keys():
        test_vals[i] = st.slider(i, min_value=num_ranges[i][0],
                                 max_value=num_ranges[i][1], key=f"{i}{key}")
    for i in cat_options.keys():
        test_vals[i] = cat_options[i][st.selectbox(
            i, options=cat_options[i].keys(), key=f"{i}{key}")]

    if st.button("Predict", key=key):
        callBack(np.array(list(test_vals.values())).reshape(1, -1))
