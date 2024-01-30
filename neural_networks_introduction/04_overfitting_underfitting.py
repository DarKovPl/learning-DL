import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
import plotly.graph_objects as go


tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

datasets = tf.keras.datasets
utils = tf.keras.utils
models = tf.keras.models
layers = tf.keras.layers
model_regularizer = tf.keras.regularizers
sns.set()

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data()

print(f"train_data shape: {x_train.shape}")
print(f"test_data shape: {x_test.shape}")

INDEX_FROM = 3
word_index = datasets.imdb.get_word_index()
word_index = {k: (v + INDEX_FROM) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
index_word = {v: k for k, v in word_index.items()}
print(" ".join(index_word[idx] for idx in x_train[0]))


def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


max_index = max([max(sequence) for sequence in np.concatenate((x_train, x_test))])

x_train = vectorize_sequences(x_train, max_index + 1)
x_test = vectorize_sequences(x_test, max_index + 1)
print(f"\n{x_train.shape}")
print(f"\n{x_test.shape}")
print("******************************************************************")

smaller_model = models.Sequential()
smaller_model.add(
    layers.Dense(units=4, activation="relu", input_shape=(max_index + 1,))
)
smaller_model.add(layers.Dense(units=4, activation="relu"))
smaller_model.add(layers.Dense(units=1, activation="sigmoid"))

smaller_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)
smaller_model.summary()

with tf.device("/CPU:0"):
    smaller_history = smaller_model.fit(
        x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test)
    )
print("******************************************************************")

baseline_line = models.Sequential()
baseline_line.add(
    layers.Dense(units=16, activation="relu", input_shape=(max_index + 1,))
)
baseline_line.add(layers.Dense(units=16, activation="relu"))
baseline_line.add(layers.Dense(units=1, activation="sigmoid"))

baseline_line.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)
baseline_line.summary()

with tf.device("/CPU:0"):
    baseline_history = baseline_line.fit(
        x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test)
    )
print("******************************************************************")

bigger_model = models.Sequential()
bigger_model.add(
    layers.Dense(units=512, activation="relu", input_shape=(max_index + 1,))
)
bigger_model.add(layers.Dense(units=512, activation="relu"))
bigger_model.add(layers.Dense(units=1, activation="sigmoid"))

bigger_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)
bigger_model.summary()

with tf.device("/CPU:0"):
    bigger_history = bigger_model.fit(
        x_train, y_train, epochs=20, batch_size=256, validation_data=(x_test, y_test)
    )
print("******************************************************************")

fig = go.Figure()
for name, history in zip(
    ["smaller", "baseline", "bigger"],
    [smaller_history, baseline_history, bigger_history],
):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["binary_crossentropy"],
            name=f"{name}_binary_crossentropy",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["val_binary_crossentropy"],
            name=f"{name}_val_binary_crossentropy",
            mode="lines+markers",
        )
    )
    fig.update_layout(xaxis_title="Epochs", yaxis_title="Binary Crossentropy")
fig.show()
print("******************************************************************")


regularized_model = models.Sequential()
regularized_model.add(
    layers.Dense(
        units=64,
        kernel_regularizer=model_regularizer.l2(0.0001),
        activation="relu",
        input_shape=(max_index + 1,),
    )
)
regularized_model.add(layers.Dropout(0.6))
regularized_model.add(
    layers.Dense(
        units=32, kernel_regularizer=model_regularizer.l2(0.01), activation="relu"
    )
)
regularized_model.add(layers.Dropout(0.4))
regularized_model.add(
    layers.Dense(
        units=16, kernel_regularizer=model_regularizer.l2(0.1), activation="relu"
    )
)
regularized_model.add(layers.Dropout(0.2))
regularized_model.add(layers.Dense(units=1, activation="sigmoid"))

regularized_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)
regularized_model.summary()

with tf.device("/CPU:0"):
    regularized_history = regularized_model.fit(
        x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test)
    )

fig = go.Figure()
for name, history in zip(
    ["baseline", "regularized"], [baseline_history, regularized_history]
):
    hist_reg = pd.DataFrame(history.history)
    hist_reg["epoch"] = history.epoch
    fig.add_trace(
        go.Scatter(
            x=hist_reg["epoch"],
            y=hist_reg["binary_crossentropy"],
            name=f"{name}_binary_crossentropy",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist_reg["epoch"],
            y=hist_reg["val_binary_crossentropy"],
            name=f"{name}_val_binary_crossentropy",
            mode="lines+markers",
        )
    )
    fig.update_layout(xaxis_title="Epochs", yaxis_title="Binary Crossentropy")
fig.show()
