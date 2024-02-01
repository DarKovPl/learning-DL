import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

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

for i in df_housing.columns:
    px.histogram(df_housing, x=i).show()

print(df_housing.info())
print('******************************************************************')

