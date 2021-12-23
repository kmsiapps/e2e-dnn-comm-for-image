import tensorflow as tf
import random

class RayleighChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
    

    def call(self, inputs):
        # power normalization
        normalizer = tf.reduce_mean(inputs)
        x = inputs / normalizer
        
        h = tf.random.normal(
            inputs.shape,
            mean=0,
            stddev=1.0
        )

        snrdB = random.randint(10, 40)
        snr = 10 ** (snrdB / 10)

        n = tf.random.normal(
            inputs.shape,
            mean=0,
            stddev=tf.math.sqrt(1/self.snr)
        )

        y = h * x + n

        yhat = y * normalizer
        yhat = tf.math.divide_no_nan(y, h)

        return yhat


class AWGNChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
    

    def call(self, inputs):
        # power normalization
        normalizer = tf.reduce_mean(inputs)
        x = inputs / normalizer

        snrdB = random.randint(10, 40)
        snr = 10 ** (snrdB / 10)

        n = tf.random.normal(
            inputs.shape,
            mean=0,
            stddev=tf.math.sqrt(1/self.snr)
        )

        y = x + n

        yhat = y * normalizer
        return yhat
