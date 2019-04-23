from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Lambda
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical

import tensorflow as tf

import csv
import ntpath
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob 
import os

batch_size = 64
epochs = 5

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

                images.append(image)
                # load steering angle
                label = int(sample[0])
                labels.append(label)   

            # yield results
            x_train = np.array(images, dtype='float32')
            y_train = np.array(labels, dtype='int32')
            y_binary = to_categorical(y_train)

            yield shuffle(x_train, y_binary)
                    
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
    model.add(Dense(4))    
    return model

################################
# MAIN
path = "/home/student/capstone/ros/src/tl_detector/data/"
samples = []
for i in range(5):
    label = i
    if i == 3:
        continue #skip class 3
    if i == 4:
        label = 3 #remap 4 (UNKNOWN)

    files = glob.glob(path + str(i) + "/*.jpg")
    for f in files:
        samples.append([label, f])

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# create generators for data
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

print("train samples: " + str(len(train_samples)))
print("valid samples: " + str(len(validation_samples)))

##todo: test with existing model
#from keras.applications.inception_v3 import InceptionV3
#model = InceptionV3(weights='imagenet', include_top=False) #v3 needs 299x299 images


model = create_model()
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size,
          validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size,
          epochs=epochs,
          verbose=1)

model.save('model.h5')

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.metrics['accuracy'])
#plt.title('model mean squared error loss')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()