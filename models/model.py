import tensorflow as tf
import tensorflow_addons as tfa

from models.resblock import ResBlock
from models.vitencoder import ViTEncoder
from models.unet_layer import Attention_UNet
from models.channellayer import RayleighChannel, AWGNChannel

class E2EImageCommunicator(tf.keras.Model):
    def __init__(self, filters=[32, 64, 128], l=4, snrdB=25, channel='Rayleigh'):
        super().__init__()

        self.encoder = tf.keras.Sequential()
        for filter in [32, 64, 64, 32]:
            self.encoder.add(ResBlock(
                filter_size=filter,
                stride=1,
                is_bottleneck=True
            ))
        
        self.conv1 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            strides=1,
            activation='sigmoid'
        )


        if channel.lower() == 'rayleigh':
            self.channel = RayleighChannel(snrdB=snrdB)
        else:
            self.channel = AWGNChannel(snrdB=snrdB)


        self.decoder = tf.keras.Sequential()
        for _ in range(l):
            self.decoder.add(ViTEncoder(
                num_heads=4,
                head_size=64,
                mlp_dim=[64, 128, 64]
            ))
        
        self.residual_proj = tf.keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=1,
            strides=1
        )

        KERNEL_SIZE = 3

        self.autoencoder = tf.keras.Sequential()
        self.autoencoder.add(tf.keras.layers.Conv2D(filters[0], KERNEL_SIZE, activation='relu', padding='same'))
        self.autoencoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
        self.autoencoder.add(tf.keras.layers.Conv2D(filters[1], KERNEL_SIZE, activation='relu', padding='same'))
        self.autoencoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))

        self.autoencoder.add(tf.keras.layers.Conv2D(filters[2], (1, 1), activation='relu', padding='same'))

        self.autoencoder.add(tf.keras.layers.Conv2DTranspose(filters[1], KERNEL_SIZE, strides=2, activation='relu', padding='same'))
        self.autoencoder.add(tf.keras.layers.Conv2DTranspose(filters[0], KERNEL_SIZE, strides=2, activation='relu', padding='same'))

        self.conv2 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            strides=1,
            activation='sigmoid'
        )

        self.mhsa = tfa.layers.MultiHeadAttention(
            head_size=filters[0] // 4,
            num_heads=4,
            output_size=filters[0]
        )
   

    def call(self, inputs, training=False):
        x = self.encoder(inputs)
        x = self.conv1(x)

        x = self.channel(x)

        x = self.decoder(x)
        x1 = self.residual_proj(x)
        x = self.autoencoder(x, training=training)

        x1 = self.mhsa([x, x1])
        x = tf.add(x, x1)
        x = self.conv2(x)

        return x


class E2E_Encoder(E2EImageCommunicator):
    def __init__(self, filters=[32, 64, 128], l=4, snrdB=25, channel='Rayleigh'):
        super().__init__(filters, l, snrdB, channel)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.conv1(x)

        return x


class E2E_Channel(E2EImageCommunicator):
    def __init__(self, filters=[32, 64, 128], l=4, snrdB=25, channel='Rayleigh'):
        super().__init__(filters, l, snrdB, channel)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.conv1(x)

        x = self.channel(x)

        return x


class E2E_Decoder(E2EImageCommunicator):
    def __init__(self, filters=[32, 64, 128], l=4, snrdB=25, channel='Rayleigh'):
        super().__init__(filters, l, snrdB, channel)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.conv1(x)

        x = self.channel(x)

        x = self.decoder(x)

        return x


class E2E_AutoEncoder(E2EImageCommunicator):
    def __init__(self, filters=[32, 64, 128], l=4, snrdB=25, channel='Rayleigh'):
        super().__init__(filters, l, snrdB, channel)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.conv1(x)

        x = self.channel(x)

        x = self.decoder(x)
        x1 = self.residual_proj(x)
        x = self.autoencoder(x)

        x = tf.add(x, x1)
        
        return x