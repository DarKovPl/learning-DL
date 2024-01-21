import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.activations import relu
from tensorflow.keras.activations import tanh
from tensorflow.keras.utils import to_categorical

np.random.seed(42)

print(tf.__version__)

model = Sequential()
print(f'\nModel:{model}')

model.add(Dense(units=4, input_shape=(10,)))
model.add(Dense(units=2))
print(f'\n{model.summary()}')
print("******************************************************************")

random_data = sorted(np.random.randn(200))

data_linear = pd.DataFrame({'data': random_data, 'linear': linear(random_data)})
print(f'\n{data_linear.head()}')
px.line(data_linear, x='data', y='linear', width=1900)

data_sigmoid = pd.DataFrame({'data': random_data, 'sigmoid': sigmoid(random_data)})
print(f'\n{data_sigmoid.head(20)}')
px.line(data_sigmoid, x='data', y='sigmoid', width=1900)

data_relu = pd.DataFrame({'data': random_data, 'relu': relu(random_data)})
print(f'\n{data_relu.head()}')
px.line(data_relu, x='data', y='relu', width=1900)

data_tanh = pd.DataFrame({'data': random_data, 'tanh': tanh(random_data)})
print(f'\n{data_tanh.head()}')
px.line(data_tanh, x='data', y='tanh', width=1900)
print("******************************************************************")
print("Binary classification")

# data = np.random.randn(1000, 150)
# labels = np.random.randint(2, size=(1000, 1))
# print(f'Data:\n{data[:2]}')
# print(f'\nLabels: \n{labels[:2]}')
# print(data.shape)
# print(labels.shape)
#
# df_data_binary = pd.DataFrame(data=np.c_[data, labels], columns=list(range(1, len(data[0]) + 1)) + ['labels'])
# print(df_data_binary.head())
#
# model = Sequential()
# model.add(Dense(units=32, activation='relu', input_shape=(len(data[0]),)))
# model.add(Dense(units=1, activation='sigmoid'))
# model.summary()
#
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(df_data_binary.iloc[:, :len(data[0])], df_data_binary['labels'], epochs=20, validation_split=0.2, verbose=1)
# history = model.fit(df_data_binary.iloc[:, :len(data[0])], df_data_binary['labels'], epochs=20, validation_split=0.2, verbose=1)
#
# metrics = history.history
# print(f'\n{metrics.keys()}')
#
# test_data = np.random.randn(5, 150)
# test_labels = np.random.randint(2, size=(5, 1))
#
# df_data_test = pd.DataFrame(
#     data=np.c_[test_data, test_labels], columns=list(range(1, len(test_data[0]) + 1)) + ['labels']
# )
#
# predicted = model.predict(df_data_test.iloc[:, :len(test_data[0])])
# predicted_class = (predicted > 0.5).astype("int32")
# print(f'\n{predicted}')
# print(predicted_class)
# print(test_labels)
print("******************************************************************")
print("Multiclass classification")

data = np.random.random((1000, 150))
labels = np.random.randint(10, size=(1000, 1))
print(f'Data shape: {data.shape}')
print(f'Data shape: {labels.shape}')

labels = to_categorical(labels, num_classes=10)

df_data_multiclass = pd.DataFrame(
    data=np.c_[data, labels], columns=list(range(1, len(data[0]) + 1)) + [f'labels_{i}' for i in range(1, 11)]
)
print(df_data_multiclass.head().to_string())

model = Sequential()
model.add(
    Dense(
        units=32, activation='relu', input_shape=(len(data[0]), )
    )
)
model.add(
    Dense(
        units=10, activation='softmax'
    )
)
model.compile(
    optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']
)
model.fit(
    df_data_multiclass.iloc[:, :len(data[0])],
    df_data_multiclass.loc[:, df_data_multiclass.columns.str.match('^labels.*', ).fillna(False)],
    batch_size=32,
    epochs=30,
    validation_split=0.2
)

test_data_multiclass = np.random.random((10, 150))
predicted = model.predict(test_data_multiclass)
predicted_multiclass = np.argmax(predicted, axis=1)
print(predicted_multiclass)
