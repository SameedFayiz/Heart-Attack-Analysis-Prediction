import streamlit as st
import numpy as np
import pandas as pd
import utils
from models.neuralNet import loadModel
from components.sidebar import viewSideBar
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from models.neuralNetTf import loadModelTf


# Page configurations
st.set_page_config(layout="wide")
viewSideBar()
TOC = utils.TableOfContent()

# -----------------------------------------------------------
st.title("Heart Attack Prediction using Neural Network")


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
X = np.array(data.drop('output', axis=1)) # Features
y = np.array(data['output']) # Target''')

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
X_train = object.fit_transform(X_train)
X_test = object.transform(X_test)
print(f"Training set -> {X_train.shape}\\nTest set ->{X_test.shape}")
''')

X_train, X_test, y_train, y_test = utils.preProcessAndSplit(df)
st.text(f"Training set -> {X_train.shape}\nTest set ->{X_test.shape}")

st.divider()


# Python implementation
st.header("Neural Network using Python", anchor=TOC.addAnchor(
    "Neural Network using Python"))
st.subheader("Neural network without Tensorflow",
             anchor=TOC.addSubAnchor("Neural Network using Python", "Neural network without Tensorflow"))
st.write("Abstract class for neural network's layer")
st.code('''
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass
''')
st.write("Fully Connected layer code")
st.code('''
class FullyConnectedLayer(Layer):
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons) - 0.5
        self.bias = np.random.rand(1, n_neurons) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output

    def backward_propagation(self, output_error, learning_rate):
        m = self.input.shape[-1]
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.reshape(-1, m).T, output_error)
        bias_error = output_error

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error

        return input_error
''')
st.write("Activation layer code")
st.code('''
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
''')
st.write("Loss & activation functions")
st.code('''
# Loss function
def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))


def BinaryCrossEntropyPrime(y_true, y_pred):
    return (y_pred - y_true) / y_true.size


# Activation functions
def Relu(inputs):
    def f(x): return (x > 0) * x
    return f(inputs)


def Relu_prime(inputs):
    def f(x): return (x > 0) * 1
    return f(inputs)


def sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))


def sigmoid_prime(inputs):
    return sigmoid(inputs) * (1-sigmoid(inputs))
''')
st.write("Network class code")
st.code('''
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.errors = []
        self.accuracy = []

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):
        result = []
        for i in range(len(input)):
            output = input[i]

            for layer in self.layers:
                output = layer.forward_propagation(output)

            result.append(output[0, 0])

        return np.array(result)

    def fit(self, X, y, epochs, learning_rate):
        errors = []
        accuracy = []
        for epoch in range(epochs):
            err = 0

            for i in range(len(X)):
                output = X[i]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y[i], output)
                error = self.loss_prime(y[i], output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= X.shape[0]

            y_pred = self.predict(X)
            accuracy.append(accuracy_score(y, (y_pred > 0.5).astype(int)))
            errors.append(err)

            if epoch % 100 == 0 or epoch == (epochs-1):
                print('epoch %d/%d   error=%f' % (epoch+1, epochs, err))
        self.errors = errors
        self.accuracy = accuracy
        return {"loss": np.array(errors), "accuracy": np.array(accuracy)}
''')
st.write("Now we make the network")
st.code('''
net = NeuralNetwork()
net.add(FullyConnectedLayer(13, 13))
net.add(ActivationLayer(Relu, Relu_prime))
net.add(FullyConnectedLayer(13, 26))
net.add(ActivationLayer(Relu, Relu_prime))
net.add(FullyConnectedLayer(26, 1))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.use(BinaryCrossEntropy, BinaryCrossEntropyPrime)
''')

# Training
st.subheader("Training the model", anchor=TOC.addSubAnchor(
    "Neural Network using Python", "Training the model"))
st.code("model_metrics = net.fit(X_train, y_train, epochs=1000, learning_rate=0.002)")

if "model" not in st.session_state:
    st.session_state["model"] = None

container = st.container(border=True)
if st.button("Start training"):
    st.session_state["model"] = utils.trainNNModel(
        X_train, y_train, container)[0]
else:
    st.session_state["model"] = loadModel()

model_metrics = {"loss": st.session_state["model"].errors,
                 "accuracy": st.session_state["model"].accuracy}
st.code('''print(f'Loss: { model_metrics["loss"][-1]: .3}\\n')
pd.DataFrame({"loss":model_metrics["loss"],"accuracy":model_metrics["accuracy"]}).plot(xlabel="epochs")''')
st.write(f'Loss: { model_metrics["loss"][-1]: .3}')

st.line_chart(pd.DataFrame(
    {"loss": model_metrics["loss"], "accuracy": model_metrics["accuracy"]}), use_container_width=False)

# Prediction
st.subheader("Prediction on Test Data",
             anchor=TOC.addSubAnchor("Neural Network using Python", "Prediction on Test Data"))
st.code('''y_pred=net.predict(X_test)
y_pred=(y_pred > 0.5).astype(int)''')
y_pred = utils.predNNModel(X_test, st.session_state["model"])
st.write(y_pred.reshape(1, -1))

# Evaluation
st.subheader("Model evalaution",
             anchor=TOC.addSubAnchor("Neural Network using Python", "Model evalaution"))
st.write("**Accuracy:**")
st.code('''
print(f'Total test size: {y_test.size}, Correct predictions: { (y_pred == y_test).sum() }\\n')

from sklearn.metrics import accuracy_score
print(f'Test accuracy: { accuracy_score(y_test,y_pred) :.3}\\n')
''')
st.write(
    f"Total test size: {y_test.size}, Correct predictions: { (y_pred == y_test).sum() }\n\nTest accuracy: { accuracy_score(y_test,y_pred) :.3}")

st.write("**Confusion matrix:**")
st.write(pd.DataFrame(confusion_matrix(y_test, y_pred), index=["Positive Predicted", "Negative Predicted"], columns=[
    "Actual Positive", "Actual Negative"]).style.apply(lambda x: x.map(utils.gradient_color), axis=None))

st.write("**Classification report:**")
report = classification_report(
    y_test, y_pred, output_dict=True, target_names=["Class 0", "Class 1"])
report.update({"accuracy": {"precision": None, "recall": None,
              "f1-score": report["accuracy"], "support": report['macro avg']['support']}})
st.dataframe(pd.DataFrame(report).T, use_container_width=True)

st.divider()


# Tensorflow implementation
st.header("Neural Network using Tensorflow", anchor=TOC.addAnchor(
    "Neural Network using Tensorflow"))
st.subheader("Model building", anchor=TOC.addSubAnchor(
    "Neural Network using Tensorflow", "Model building"))
st.write("**Importing necessary libraries**")
st.code('''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense''')
st.write("**Model architecture**")
st.code('''
model = Sequential([
    Dense(13, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(26, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
''')

# Training
st.subheader("Training Tensorflow model", anchor=TOC.addSubAnchor(
    "Neural Network using Tensorflow", "Training Tensorflow model"))
st.code('''
history = model.fit(X_train, y_train, epochs=1000, verbose=0)
pd.DataFrame(history.history).plot(xlabel="epochs");''')

if "modelTf" not in st.session_state:
    st.session_state["modelTf"] = None

st.session_state["modelTf"], history = loadModelTf()
st.line_chart(history.drop(
    history.columns[0], axis=1), use_container_width=False)
st.write("**Model summary**")
st.code('''
model.summary()''')
st.session_state["modelTf"].summary(print_fn=lambda x: st.text(x))

# Prediction
st.subheader("Predictions on Test Data",
             anchor=TOC.addSubAnchor("Neural Network using Tensorflow", "Predictions on Test Data"))
st.code('''
y_pred_tf = model.predict(X_test, verbose=0)
y_pred_tf = (y_pred_tf > 0.5).astype(int)''')

y_pred_tf = st.session_state["modelTf"].predict(X_test, verbose=0)
y_pred_tf = (y_pred_tf > 0.5).astype(int)
st.write(y_pred_tf.T)
y_pred_tf = y_pred_tf.ravel()

# Evaluation
st.subheader("Tensorflow model evalaution",
             anchor=TOC.addSubAnchor("Neural Network using Tensorflow", "Tensorflow model evalaution"))

st.write("**Accuracy:**")
st.code('''
print(f'Total test size: {y_test.size}, Correct predictions: { (y_pred_tf == y_test).sum() }\\n')

from sklearn.metrics import accuracy_score
print(f'Test accuracy: { accuracy_score(y_test,y_pred_tf) :.3}\\n')
''')
st.write(
    f"Total test size: {y_test.size}, Correct predictions: { (y_pred_tf == y_test).sum() }\n\nTest accuracy: { accuracy_score(y_test,y_pred_tf) :.3}")

st.write("**Confusion matrix:**")
st.write(pd.DataFrame(confusion_matrix(y_test, y_pred_tf), index=["Positive Predicted", "Negative Predicted"], columns=[
    "Actual Positive", "Actual Negative"]).style.apply(lambda x: x.map(utils.gradient_color), axis=None))

st.write("**Classification report:**")
report = classification_report(
    y_test, y_pred_tf, output_dict=True, target_names=["Class 0", "Class 1"])
report.update({"accuracy": {"precision": None, "recall": None,
              "f1-score": report["accuracy"], "support": report['macro avg']['support']}})
st.dataframe(pd.DataFrame(report).T, use_container_width=True)

st.divider()

TOC.genTableOfContent()
