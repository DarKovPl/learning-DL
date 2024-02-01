import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tensorflow as tf

datasets = tf.keras.datasets
utils = tf.keras.utils
models = tf.keras.models
layers = tf.keras.layers
kernel_regularizer = tf.keras.regularizers

np.set_printoptions(precision=12, suppress=True, linewidth=150)
pd.options.display.float_format = "{:.6f}".format
sns.set()

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("******************************************************************")

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

scale = StandardScaler() #MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scale.fit_transform(x_train.reshape(x_train.shape[0], -1))
x_test_scaled = scale.transform(x_test.reshape(x_test.shape[0], -1))
print(f"x_train_scaled shape: {x_train_scaled.shape}")
print(f"x_test_scaled shape: {x_test_scaled.shape}")
print("******************************************************************")

model = models.Sequential()
model.add(
    layers.Dense(
        units=512,
        activation="relu",
        kernel_regularizer=kernel_regularizer.l2(0.000001),
        input_shape=(x_train_scaled.shape[1],),
    )
)
model.add(layers.Dropout(0.3))
model.add(
    layers.Dense(
        units=256, kernel_regularizer=kernel_regularizer.l2(0.00001), activation="relu"
    )
)
model.add(layers.Dropout(0.3))
model.add(
    layers.Dense(
        units=128, kernel_regularizer=kernel_regularizer.l2(0.0001), activation="relu"
    )
)
model.add(layers.Dropout(0.3))
model.add(
    layers.Dense(
        units=64, kernel_regularizer=kernel_regularizer.l2(0.0001), activation="relu"
    )
)
model.add(layers.Dense(units=10, activation="softmax"))
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()

history = model.fit(
    x_train_scaled,
    y_train,
    epochs=30,
    batch_size=50,
    validation_split=0.3,
)
print("******************************************************************")

metrics = pd.DataFrame(history.history)
metrics["epoch"] = history.epoch
print(metrics.head())

fig = make_subplots(rows=1, cols=2)
fig.add_trace(
    go.Scatter(
        x=metrics["epoch"],
        y=metrics["accuracy"],
        name="accuracy",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=metrics["epoch"], y=metrics["loss"], name="loss"), row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=metrics["epoch"],
        y=metrics["val_accuracy"],
        name="val_accuracy",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=metrics["epoch"], y=metrics["val_loss"], name="val_loss"), row=1, col=2
)
fig.update_xaxes(title_text="Epoch")
fig.update_yaxes(title_text="Accuracy")
fig.update_layout(
    title="Training History",
    xaxis_title="Epoch",
    yaxis_title="Accuracy",
    template="plotly_white",
)
fig.show()
print(f"Evaluate: {model.evaluate(x_test_scaled, y_test, verbose=2)}")
print("******************************************************************")

predictions = model.predict(x_test_scaled)
predictions_df = pd.DataFrame(predictions)
predictions_df["prediction"] = predictions_df.idxmax(axis=1)
predictions_df["actual"] = y_test
predictions_df["correct"] = predictions_df["prediction"] == predictions_df["actual"]
print((predictions_df["correct"] == True).sum() / predictions_df.shape[0])
print(predictions_df.head(10))
print("******************************************************************")
