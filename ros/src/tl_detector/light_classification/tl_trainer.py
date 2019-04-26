from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical

import tensorflow as tf

import csv
import ntpath
import numpy as np
import cv2
import glob 
import os

batch_size = 64
epochs = 10

PATH = "../data/"
NUM_CLASSES = 3

INCEPTION = True

# input image dimensions
input_shape = (300, 400, 3)
          
def generator(samples, batch_size=32):
    n_samples = len(samples)
    while(1):
        shuffle(samples)
        #iterate through all batches
        for i in range(0, n_samples, batch_size):
            # grab batch
            batch_samples = samples[i:i+batch_size]
            # load data from HDD
            images = []
            labels = []
            for sample in batch_samples:
                # load image
                image = cv2.imread(sample[1])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (400,300), interpolation = cv2.INTER_CUBIC)                 
                image = image[0:299, 0:299] #crop
                images.append(image)
                
                #flip horizontally
                images.append(cv2.flip(image, 0))                
                
                # load steering angle
                label = int(sample[0])
                labels.append(label)   
                labels.append(label)  #the same label for the flipped image

            # yield results
            x = np.array(images, dtype='float32')            
            y = np.array(labels, dtype='int32')
            y_binary = to_categorical(y)

            yield shuffle(x, y_binary)
                    
def create_model():
    model = Sequential()
    # preproc
    #model.add(Cropping2D(cropping=((20,20), (0,30)), input_shape=input_shape))
    model.add(Lambda(lambda x: tf.image.rgb_to_hsv(x), input_shape=input_shape))
    model.add(Lambda(lambda x: x[:,:,:,0][:,:,:,np.newaxis]))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    # convolutions
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # flatten
    model.add(Dropout(0.25))
    model.add(Flatten())
    # dense
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    # output
    model.add(Dense(NUM_CLASSES))    
    return model

def create_inception():    
    from keras.applications.inception_v3 import InceptionV3# create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    out = Dense(NUM_CLASSES, activation='softmax')(x)    
    model = Model(inputs=base_model.input, outputs=out)
    
    return base_model, model

def train_model(base_model):    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size,
              validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size,
              epochs=epochs,
              verbose=1)

def train_inception(base_model, model):
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])

    model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size,
              validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size,
              epochs=epochs,
              verbose=1)
    
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True# we need to recompile the model for these modifications to take effect
    
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size,
              validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size,
              epochs=epochs,
              verbose=1)   

################################
# MAIN
samples = []
for i in range(3):
    label = i
    if i == 3:
        continue #skip class 3
    if i == 4:
        label = 3 #remap 4 (UNKNOWN)

    files = glob.glob(PATH + str(i) + "/*.jpg")
    for f in files:
        samples.append([label, f])

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# create generators for data
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

print("train samples: " + str(len(train_samples)))
print("valid samples: " + str(len(validation_samples)))

model = None
if INCEPTION:
    base, model = create_inception()  
    train_inception(base, model)    
else:
    model = create_model()
    train_model(model)   

model.save('model.h5')


    