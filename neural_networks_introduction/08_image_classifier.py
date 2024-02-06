import pandas as pd
import tensorflow as tf
import os
import plotly.graph_objects as go
from datetime import datetime as dt

image = tf.keras.preprocessing.image
applications = tf.keras.applications
utils = tf.keras.utils
models = tf.keras.models
layers = tf.keras.layers
optimizers = tf.keras.optimizers
kernel_regularizer = tf.keras.regularizers

base_dir = r'../images/datasets/data/planes'
raw_no_of_files = {}
classes = ['drone', 'fighter-jet', 'helicopter', 'missile', 'passenger-plane', 'rocket']
for directory in classes:
    raw_no_of_files[directory] = len(os.listdir(os.path.join(base_dir, directory)))

print(raw_no_of_files.items())
print("******************************************************************")

main_dir = r'../images/datasets/'
train_dir = os.path.join(main_dir, 'train')
valid_dir = os.path.join(main_dir, 'valid')
test_dir = os.path.join(main_dir, 'test')

train_data_gen = image.ImageDataGenerator(
    rotation_range=40,
    rescale=1. / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_data_gen = image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_data_gen.flow_from_directory(
    directory=train_dir, target_size=(150, 150), class_mode='binary', batch_size=20
)

validation_generator = validation_data_gen.flow_from_directory(
    directory=valid_dir, target_size=(150, 150), class_mode='binary', batch_size=20
)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(512, kernel_regularizer=kernel_regularizer.l2(1e-6), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, kernel_regularizer=kernel_regularizer.l2(1e-4), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

log_dir = f"../logs/fit/{dt.now().strftime('%Y%m%d-%H%M%S')}"

batch_size = 32
steps_per_epoch = 919 // batch_size
validation_steps = 262 // batch_size
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# history = model.fit_generator(generator=train_generator,
#                               steps_per_epoch=steps_per_epoch,
#                               epochs=30,
#                               validation_data=validation_generator,
#                               validation_steps=validation_steps,
#                               callbacks=[tensorboard])
#


def plot_hist(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='accuracy', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='val_accuracy', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='Accuracy vs. Val Accuracy', xaxis_title='Epoki',
                      yaxis_title='Accuracy', yaxis_type='log')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='loss', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='val_loss', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='Loss vs. Val Loss', xaxis_title='Epoki', yaxis_title='Loss',
                      yaxis_type='log')
    fig.show()


def launch_tensorboard():
    os.system(f"tensorboard --logdir={log_dir}")


# launch_tensorboard()

print("******************************************************************")

conv_base = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, kernel_regularizer=kernel_regularizer.l2(1e-4), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

print(f'Number of layers trained before freezing {len(model.trainable_weights)}')
conv_base.trainable = False
print(f'Number of layers trained after freezing {len(model.trainable_weights)}')

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=50,  # 100
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              callbacks=[tensorboard])

plot_hist(history)
print("******************************************************************")

