import tensorflow as tf
from keras import layers

class DeepModel(tf.keras.Model): # Ensure this name matches
    def __init__(self):
        super(DeepModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        # ... your model code ...