import tensorflow as tf
import tensorflow_addons as tfa

class MLP(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(
            hidden_features
        )
        self.gelu = tf.nn.gelu
        self.fc2 = tf.keras.layers.Dense(
            out_features
        )
    
    
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.gelu(x)
        x = self.fc2(x)

        return x


class ViTEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads=16, head_size=4, mlp_dim=[64, 128, 32]):
        super().__init__()

        self.ln1 = tf.keras.layers.LayerNormalization()

        self.resweight1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=1,
            padding='same'
        )

        self.mhsa = tfa.layers.MultiHeadAttention(
            head_size=head_size,
            num_heads=num_heads,
            output_size=64
        )

        self.resweight2 = tf.keras.layers.Conv2D(
            filters=mlp_dim[0],
            kernel_size=1,
            padding='same'
        )

        self.ln2 = tf.keras.layers.LayerNormalization()

        self.mlp = MLP(*mlp_dim)
    

    def call(self, inputs):
        x = self.ln1(inputs)
        x_residual = self.resweight1(x)
        x = self.mhsa([x, x])
        x = tf.add(x, x_residual)
        
        x_residual = self.resweight2(x)
        x = self.ln2(x)
        x = self.mlp(x)
        x = tf.add(x, x_residual)

        return x