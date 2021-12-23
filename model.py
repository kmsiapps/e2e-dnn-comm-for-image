import tensorflow as tf

from resblock import ResBlock
from channellayer import RayleighChannel, AWGNChannel
from config import SNRDB

class E2EImageCommunicator(tf.keras.Model):
    def __init__(self, l=4, snrdB=SNRDB, channel='Rayleigh'):
        super().__init__()
        
        '''
        self.encoder = tf.keras.Sequential()
        for _ in range(l):
            self.encoder.add(ViTEncoder(
                num_heads=16,
                head_size=4,
                mlp_dim=[64, 128, 32]
            ))
        '''
        self.encoder = tf.keras.Sequential()
        for _ in range(l):
            self.encoder.add(ResBlock(
                filter_size=32,
                stride=1,
                is_bottleneck=True
            ))

        if channel.lower() == 'rayleigh':
            self.channel = RayleighChannel(snrdB=snrdB)
        else:
            self.channel = AWGNChannel(snrdB=snrdB)

        '''
        self.decoder = tf.keras.Sequential()
        for _ in range(l):
            self.decoder.add(ViTEncoder(
                num_heads=4,
                head_size=64,
                mlp_dim=[64, 128, 64]
            ))
        '''
        self.decoder = tf.keras.Sequential()
        for _ in range(l):
            self.decoder.add(ResBlock(
                filter_size=32,
                stride=1,
                is_bottleneck=True
            ))

        # self.ln = tf.keras.layers.LayerNormalization()
        
        self.conv = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            padding='same'
        )

        # self.relu = tf.nn.relu
    

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.channel(x)
        x = self.decoder(x)
        # x = self.ln(x)
        x = self.conv(x)
        # x = self.relu(x)

        return x


class E2E_Encoder(E2EImageCommunicator):
    def __init__(self, l=4, snrdB=SNRDB, channel='Rayleigh'):
        super().__init__(l, snrdB, channel)

    def call(self, inputs):
        x = self.encoder(inputs)
        return x


class E2E_Channel(E2EImageCommunicator):
    def __init__(self, l=4, snrdB=SNRDB, channel='Rayleigh'):
        super().__init__(l, snrdB, channel)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.channel(x)
        return x


class E2E_Decoder(E2EImageCommunicator):
    def __init__(self, l=4, snrdB=SNRDB, channel='Rayleigh'):
        super().__init__(l, snrdB, channel)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.channel(x)
        x = self.decoder(x)
        return x
