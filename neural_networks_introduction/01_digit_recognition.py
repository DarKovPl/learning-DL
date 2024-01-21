import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.python.client import device_lib

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.set_printoptions(precision=12, suppress=True, linewidth=120)

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

(x_train, y_train), (x_test, y_test) = load_data()

print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_test shape: {y_test.shape}')
print(x_train[0])

plt.imshow(x_train[0], cmap='gray_r')
plt.axis('off')

plt.figure(figsize=(15, 15))
for i in range(1, 11):
    plt.subplot(1, 10, i)
    plt.axis('off')
    plt.imshow(x_train[i - 1], cmap='gray_r')
    plt.title(y_train[i - 1], color='black', fontsize=16)

x_train_reshaped = x_train.reshape(len(x_train), -1)
x_test_reshaped = x_test.reshape(len(x_test), -1)
print(f'x_train shape: {x_train_reshaped.shape}')
print(f'x_test shape: {x_test_reshaped.shape}')

scaler = MinMaxScaler(feature_range=(-1, 1))
x_train_reshaped = scaler.fit_transform(x_train_reshaped)
x_test_reshaped = scaler.transform(x_test_reshaped)
print(x_train_reshaped[0])

initializer = GlorotNormal(seed=42)

model = Sequential()
model.add(
    Dense(
        units=64,
        activation='tanh',
        kernel_regularizer=regularizers.l2(0.0001),
        input_shape=(784,),
        kernel_initializer=initializer
    )
)
model.add(Dropout(0.2))
model.add(
    Dense(
        units=128,
        activation='tanh',
        kernel_regularizer=regularizers.l2(0.0001),
        input_shape=(784,),
        kernel_initializer=initializer
    )
)
model.add(Dropout(0.2))
model.add(
    Dense(
        units=64,
        activation='tanh',
        kernel_regularizer=regularizers.l2(0.0001),
        input_shape=(784,),
        kernel_initializer=initializer
    )
)
model.add(Dropout(0.2))
model.add(
    Dense(
        units=32,
        activation='tanh',
        kernel_regularizer=regularizers.l2(0.0001),
        input_shape=(784,),
        kernel_initializer=initializer
    )
)
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='Adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train_reshaped, y_train, epochs=350)
print(f'\nEvaluate: {model.evaluate(x_test_reshaped, y_test, verbose=2)}\n')

metrics = pd.DataFrame(history.history)
fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Scatter(y=metrics['loss'], name='loss'), row=1, col=1)
fig.add_trace(go.Scatter(y=metrics['accuracy'], name='accuracy'), row=1, col=2)
fig.update_layout(height=1000, width=1900)
fig.show()

y_pred = model.predict(x_test_reshaped)
y_pred_class = np.argmax(y_pred, axis=1)

pred = pd.concat([pd.DataFrame(y_test, columns=['y_test']), pd.DataFrame(y_pred_class, columns=['y_pred'])], axis=1)

misclassified = pred[pred['y_test'] != pred['y_pred']]
print(misclassified.count())
