import tensorflow as tf

from models.attention_unet_blocks import ContractingBlock, ExpansiveBlock, ConvBlock


class Attention_UNet(tf.keras.layers.Layer):
    def __init__(self, filters=[16, 32, 64, 128, 256]):
        '''
        Vanilla UNet. filters: # of channels in each contracting/expanding path.
        e.g., for [16, 32, 64, 128, 256]:
        (Contracting path) 16 -> 32 -> 64 -> 128 -> 256
        (Expanding path)   256 -> 128 -> 64 -> 32-> 16
        '''
        super().__init__()

        assert(len(filters) > 0, "filters should have at least one element")

        conv_filters = filters[0]
        self.convblock1 = ConvBlock([conv_filters, conv_filters])

        self.contract_layers = []

        contracting_path_filters = filters[1:]
        for num_channel in contracting_path_filters:
            self.contract_layers.append(ContractingBlock([num_channel, num_channel]))
        
        self.expand_layers = []

        expanding_path_filters = filters[:]
        expanding_path_filters.reverse()
        expanding_path_filters = expanding_path_filters[1:]

        for num_channel in expanding_path_filters:
            self.expand_layers.append(ExpansiveBlock([num_channel, num_channel, num_channel]))

        self.conv = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            strides=1,
            activation='sigmoid'
        )


    def call(self, inputs):
        x = self.convblock1(inputs)

        residuals = []
        for contracting_block in self.contract_layers:
            residuals.append(x)
            x = contracting_block(x)

        for expanding_block in self.expand_layers:
            r = residuals.pop()
            x = expanding_block(x, r)

        output = self.conv(x)
        return output
