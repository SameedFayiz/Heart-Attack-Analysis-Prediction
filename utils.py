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

    return X_train, X_test, y_train, y_test


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
