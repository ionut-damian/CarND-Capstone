from styx_msgs.msg import TrafficLight

import shutil
import tensorflow as tf
from keras.models import load_model
import h5py
import keras
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self):        
        # check that model Keras version is same as local Keras version
        filename = './light_classification/model.h5'

        f = h5py.File(filename, mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras.__version__).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version ', keras_version,
                ', but the model was built using ', model_version)

        self.model = load_model(filename, custom_objects={"tf": tf})
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = cv2.resize(image, (400,300), interpolation = cv2.INTER_CUBIC)             
        image = image[0:299, 0:299] #crop

        image_array = np.asarray(image)

        with self.graph.as_default():
            result = self.model.predict(image[None, :, :, :], batch_size=1).squeeze()
            id = np.argmax(result)
            return id, result[id]