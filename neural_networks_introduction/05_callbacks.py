import os

import numpy as np
import pandas as pd
import tensorflow as tf
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime as dt


np.set_printoptions(precision=12, suppress=True, linewidth=120)
print(tf.__version__)

datasets = tf.keras.datasets
utils = tf.keras.utils
models = tf.keras.models
layers = tf.keras.layers
kernel_regularizer = tf.keras.regularizers
checkpoint = tf.keras.callbacks.ModelCheckpoint

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("******************************************************************")

scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train.reshape(x_train.shape[0], -1))
x_test_scaled = scale.transform(x_test.reshape(x_test.shape[0], -1))
# y_train = utils.to_categorical(y_train, num_classes=10)
print(f"X_train shape: {x_train_scaled.shape}")
print(f"X_test shape: {x_test_scaled.shape}")


def create_model():
    model = models.Sequential()
    model.add(
        layers.Dense(
            units=512,
            activation="relu",
            kernel_regularizer=kernel_regularizer.l2(0.000001),
            input_shape=(x_train_scaled.shape[1],),
        )
    )
    model.add(layers.Dropout(0.2))
    model.add(
        layers.Dense(
            units=256,
            activation="relu",
            kernel_regularizer=kernel_regularizer.l2(0.00001),
        )
    )
    model.add(layers.Dropout(0.2))
    model.add(
        layers.Dense(
            units=128,
            activation="relu",
            kernel_regularizer=kernel_regularizer.l2(0.0001),
        )
    )
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=10, activation="softmax"))
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


filepath = "../models/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = checkpoint(
    filepath=filepath,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)
build_model = create_model()
history = build_model.fit(
    x_train_scaled,
    y_train,
    epochs=20,
    validation_split=0.2,
    batch_size=512,
    callbacks=[checkpoint],
)


metrics = pd.DataFrame(history.history)
fig = make_subplots(rows=1, cols=1)
fig.add_trace(
    go.Scatter(
        x=metrics.index,
        y=metrics["accuracy"],
        name="accuracy",
        mode="lines+markers",
    ),
    row=1,
    col=1,
)
fig.update_layout(width=1900)
fig.show()
print("******************************************************************")

log_dir = f"../logs/fit/{dt.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
build_model = create_model()
history = build_model.fit(
    x_train_scaled,
    y_train,
    epochs=20,
    validation_split=0.2,
    batch_size=512,
    callbacks=[tensorboard],
)


def launch_tensorboard():
    os.system(f"tensorboard --logdir={log_dir}")


launch_tensorboard()
