# Import modules
print("hi")
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
import os

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

############################ MODEL ###################################
#Inception V3
INPUT_SHAPE = (150,150,3)

local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = INPUT_SHAPE,
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False
# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Create bottom layers
from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(81, activation='softmax')(x)

model = Model( pre_trained_model.input, x)

model.compile(loss='categorical_crossentropy',optimizer = RMSprop(lr=0.0001),metrics=['accuracy'])

# read csv with the predictions
train = pd.read_csv('train_labels.csv')
# look at the data format
print(train.head())

################################## preprocessing ################################

# perform one hot encoding
y=train['label'].values
y = to_categorical(y)
print(y)

# get validation set
X_train, X_test, y_train, y_test = train_test_split(train["img_name"], y, random_state=42, test_size=0.2)

# read the images and get them in an array
def gen_train():
    while True:
        for x, y in zip(X_train, y_train):
            img = image.load_img("train_set/train_set/"+x, target_size=INPUT_SHAPE, grayscale=False)
            img = image.img_to_array(img)
            img = img/255
            yield (np.array([img]),np.array([y]))

# read the images and get them in an array
def gen_validation():
    while True:
        for x,y in zip(X_test, y_test):
            img = image.load_img("train_set/train_set/"+x, target_size=INPUT_SHAPE, grayscale=False)
            img = image.img_to_array(img)
            img = img/255
            yield (np.array([img]),np.array([y]))

################################# Train #####################################
model.fit_generator(gen_train(), steps_per_epoch=len(X_train), epochs=10, verbose=1, validation_data=gen_validation(), validation_steps=len(X_test))
