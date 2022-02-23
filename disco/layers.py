import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow import keras as tfk
from IPython import embed


class SelfAttentionBlock(tfk.layers.Layer):
    def __init__(self, attention_dims, num_heads, ff_dims=None):
        super().__init__()
        attention_dims = attention_dims
        if ff_dims is None:
            ff_dims = attention_dims
        self.ff_dims = ff_dims

        self.att = tfk.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = attention_dims,
            value_dim = attention_dims,
            output_shape = attention_dims,
        )
        self.ff = tfk.Sequential([
            tfk.layers.Dense(ff_dims, activation='ReLU', kernel_initializer='identity'),
            tfk.layers.Dense(attention_dims, kernel_initializer='identity'),
        ])
        self.layer_norm = tfk.layers.LayerNormalization()

    def build(self, shapes):
        mask = shapes[-1]
        self.reflections_per_image = mask[-1]

    def call(self, inputs):
        qkv, mask = inputs

        attention_mask = mask@tf.transpose(mask, [0, 2, 1])

        out = self.att(qkv, qkv, qkv, attention_mask)
        out = self.layer_norm(out + qkv)
        out = self.layer_norm(self.ff(out) + out)
        out = out*mask
        return out, mask

class Assigner(tfk.models.Model):
    def __init__(self, attention_blocks, attention_dims, num_heads, ff_dims=None,  hmax=50):
        super().__init__()
        if ff_dims is None:
            ff_dims = attention_dims
        self.normalizers = None
        self.embed = tfk.layers.Dense(attention_dims, kernel_initializer='identity')
        self.normalize = tfk.layers.LayerNormalization(axis=-2)

        self.encoder_layers = []
        for i in range(attention_blocks):
            self.encoder_layers.append(SelfAttentionBlock(attention_dims,  num_heads, ff_dims=ff_dims))

        self.decoder_layers = []
        self.decoder_layers.append(tfk.layers.Dense(3 * (2*hmax + 1)))
        self.decoder_layers.append(tfk.layers.Reshape((-1, 3, 2*hmax+1)))
        self.decoder_layers.append(tfk.layers.Softmax(axis=-1))

    def call(self, inputs):
        """
        inputs : (xypos, mask)
        """
        qkv, mask = inputs

        #Preprocess qkv a bit
        qkv = self.normalize(qkv)
        qkv = self.embed(qkv)

        out = (qkv, mask)
        for layer in self.encoder_layers:
            out = layer(out)

        out = out[0]
        for layer in self.decoder_layers:
            out = layer(out)
        return out

