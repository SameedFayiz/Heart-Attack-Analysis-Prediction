import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def saveModelTf():
    data = pd.read_csv("./dataset/heart.csv")

    # data = data.drop_duplicates()
    X = np.array(data.drop('output', axis=1))  # Features
    y = np.array(data['output'])  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    object = StandardScaler()
    X_train = object.fit_transform(X_train)
    X_test = object.transform(X_test)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(13, activation='relu',
                              input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(26, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=1000, verbose=2)
    history = pd.DataFrame(history.history)
    history.to_csv("./models/static/history_tf.csv")

    model.save("./models/static/neural_net_tf.keras")


def loadModelTf():
    loadedModel = tf.keras.models.load_model(
        "./models/static/neural_net_tf.keras")
    history = pd.read_csv("./models/static/history_tf.csv")
    return loadedModel, history
