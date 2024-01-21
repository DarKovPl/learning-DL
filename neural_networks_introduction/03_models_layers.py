import numpy as np
import plotly.express as px
import json

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

mnist = tf.keras.datasets.mnist
models = tf.keras.models
layers = tf.keras.layers
utils = tf.keras.utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f'X_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {x_test.shape}')
print(f'y_test shape: {y_test.shape}')

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test_scaled = scaler.transform(x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
y_train = utils.to_categorical(y_train, num_classes=10)
print(f'X_train shape: {x_train_scaled.shape}')
print(f'X_test shape: {x_test_scaled.shape}')

model = models.Sequential()
model.add(
    layers.Dense(units=128, input_shape=(x_train_scaled.shape[1], ), activation='relu')
)
model.add(
    layers.Dropout(0.2)
)
model.add(
    layers.Dense(units=10, activation='softmax')
)

model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
)

model.summary()
model.fit(
    x_train_scaled, y_train, epochs=10, validation_split=0.25, batch_size=20
)

print(f'\nLayers: {model.layers}')
print(f'\nInputs: {model.inputs}')
print(f'\nOutputs: {model.outputs}')
print(f'\nget_config(): {model.get_config()}')
print(f'\nget_weights(): {model.get_weights()}')
print(f'\nget_weights()[0].shape: {model.get_weights()[0].shape}')
print(f'\nget_weights()[1].shape: {model.get_weights()[1].shape}')
print(f'\nget_weights()[2].shape: {model.get_weights()[2].shape}')
print("******************************************************************")

model_json = model.to_json()
parsed = json.loads(model_json)
print(json.dumps(parsed, indent=1))

model_from_json = models.model_from_json(model_json)
model_from_json.summary()
