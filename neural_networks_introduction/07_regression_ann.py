import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=12, suppress=True, linewidth=150)
pd.options.display.float_format = '{:.6f}'.format

datasets = tf.keras.datasets
utils = tf.keras.utils
models = tf.keras.models
layers = tf.keras.layers
kernel_regularizer = tf.keras.regularizers

raw_data = pd.read_csv('../datasets/housing.csv')
print(raw_data.head())
print(raw_data.info())
print(raw_data.describe())
print(raw_data.isnull().sum())
print('******************************************************************')

df_housing = raw_data.copy()
df_housing.dropna(inplace=True)
print(df_housing.isnull().sum())

print(df_housing.describe())
print(df_housing.describe(include='object').T)

# for i in df_housing.columns:
#     px.histogram(df_housing, x=i).show()

index = df_housing[
    (df_housing['median_house_value'] > 500000) | (df_housing['housing_median_age'] > 51)
    ].index
print(index)
df_housing.drop(index, inplace=True)

# for i in df_housing.columns:
#     px.histogram(df_housing, x=i).show()

print(df_housing.info())
print('******************************************************************')

df_housing_dummy = pd.get_dummies(df_housing, drop_first=True, dtype='int')
print(df_housing_dummy.head())
print(df_housing_dummy.info())
print('******************************************************************')

train_data = df_housing_dummy.sample(frac=0.75, random_state=42)
test_data = df_housing_dummy.drop(train_data.index)
print(train_data.shape)
print(test_data.shape)
print(train_data.head().to_markdown())
print(test_data.head().to_markdown())
print('******************************************************************')

train_labels = train_data.pop('median_house_value')
test_labels = test_data.pop('median_house_value')

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)
print('******************************************************************')

model = models.Sequential()
model.add(layers.Dense(
    units=1024,
    activation='gelu',
    kernel_regularizer=kernel_regularizer.l2(0.00001),
    input_shape=(train_data_scaled.shape[1],)
)
)
model.add(layers.Dropout(0.3))
model.add(layers.Dense(units=512, activation='relu', kernel_regularizer=kernel_regularizer.l2(0.00001)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(units=256, activation='gelu', kernel_regularizer=kernel_regularizer.l2(0.0001)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(units=128, activation='relu', kernel_regularizer=kernel_regularizer.l2(0.001)))
model.add(layers.Dense(units=64, activation='gelu', kernel_regularizer=kernel_regularizer.l2(0.001)))
model.add(layers.Dense(units=32, activation='relu', kernel_regularizer=kernel_regularizer.l2(0.001)))
model.add(layers.Dense(units=16, activation='gelu', kernel_regularizer=kernel_regularizer.l2(0.001)))
model.add(layers.Dense(units=1))
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
model.summary()
print('******************************************************************')

history = model.fit(
    train_data_scaled,
    train_labels,
    epochs=210,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
print('******************************************************************')


def plot_hist(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist['rmse'] = np.sqrt(hist['mse'])
    hist['val_rmse'] = np.sqrt(hist['val_mse'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['mae'], name='mae', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_mae'], name='val_mae', mode='markers+lines'))
    fig.update_layout(width=1900, height=900, title='MAE vs. VAL_MAE', xaxis_title='Epochs',
                      yaxis_title='Mean Absolute Error', yaxis_type='log').show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['rmse'], name='rmse', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_rmse'], name='val_rmse', mode='markers+lines'))
    fig.update_layout(width=1900, height=900, title='RMSE vs. VAL_RMSE', xaxis_title='Epochs',
                      yaxis_title='Root Mean Squared Error', yaxis_type='log').show()



plot_hist(history)
print('******************************************************************')

test_predictions = model.predict(test_data_scaled).flatten()
pred = pd.DataFrame({'test_labels': test_labels, 'test_predictions': test_predictions})
print(pred.head())

fig = px.scatter(pred, x='test_labels', y='test_predictions')
fig.add_trace(go.Scatter(x=[0, 550000], y=[0, 550000], mode='lines', name='lines'))
fig.show()

error = pred['test_labels'] - pred['test_predictions']
fig = px.histogram(error, nbins=100)
fig.update_layout(width=1900, height=900, title='Error Distribution', xaxis_title='Error')
fig.show()
