import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()

        # filters: # of filters list. e.g., [64 64]
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=3,
            strides=1,
            padding='same'
        )

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters[1],
            kernel_size=3,
            strides=1,
            padding='same'
        )

        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)

        return x


class ContractingBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()

        # filters: # of filters list. e.g., [64 64]
        self.maxpool = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=2,
            padding='same'
        )

        self.convblock = ConvBlock(filters)

    def call(self, inputs):
        x = self.maxpool(inputs)
        x = self.convblock(x)
        return x


class ExpansiveBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()

        # filters: # of filters list. First filter denotes 1/2 of the concatenated channels.
        # e.g., [512 512 512]
        self.upconv = tf.keras.layers.Conv2DTranspose(
            filters=filters[0],
            kernel_size=2,
            strides=2,
            padding='same'
        )

        self.convblock = ConvBlock(filters[1:])

    def call(self, inputs, concat_inputs):
        x = self.upconv(inputs)

        # Copy and crop
        _, h, w, _ = concat_inputs.shape
        _, target_h, target_w, _ = x.shape

        h_crop = (h - target_h) // 2
        w_crop = (w - target_w) // 2

        concat_x = tf.keras.layers.Cropping2D(
            cropping=((h_crop, h_crop), (w_crop, w_crop))
        )(concat_inputs)

        x = tf.concat([concat_x, x], axis=-1)
        x = self.convblock(x)
        return x
