import numpy as np
import reciprocalspaceship as rs
import tensorflow as tf
from tensorflow import keras as tfk
from IPython import embed


class ResNetLayer(tfk.layers.Layer):
    def __init__(self, 
        units, 
        dropout=None, 
        activation='ReLU',
        kernel_initializer='glorot_normal', 
        normalize=False, 
        **kwargs
        ):
        super().__init__()

        self.units = units
        self.kernel_initializer = kernel_initializer

        if dropout is not None:
            self.dropout = tf.keras.layers.Dropout(dropout)
        else:
            self.dropout = None

        if normalize:
            self.normalize_1 = tfk.layers.LayerNormalization(axis=-2)
            self.normalize_2 = tfk.layers.LayerNormalization(axis=-2)
        else:
            self.normalize_1 = None
            self.normalize_2 = None

        self.activation = tfk.activations.get(activation)

    def build(self, shape, **kwargs):
        self.ff1 = tf.keras.layers.Dense(self.units, kernel_initializer=self.kernel_initializer)
        self.ff2 = tf.keras.layers.Dense( shape[-1], kernel_initializer=self.kernel_initializer)
        
    def call(self, X, mask=None, **kwargs):
        out = X

        if self.normalize_1 is not None:
            out = self.normalize_1(out)

        out = self.activation(out)

        out = self.ff1(out)

        if self.normalize_2 is not None:
            out = self.normalize_2(out)

        out = self.activation(out)

        out = self.ff2(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = out + X
        return out

class SelfAttentionBlock(tfk.layers.Layer):
    def __init__(self, attention_dims, num_heads, ff_dims=None, kernel_initializer='glorot_normal', normalize=True):
        super().__init__()
        attention_dims = attention_dims
        if ff_dims is None:
            ff_dims = attention_dims
        self.ff_dims = ff_dims
        self.num_heads=num_heads

        self.att = tfk.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = attention_dims,
            value_dim = attention_dims,
            output_shape = attention_dims,
            kernel_initializer = kernel_initializer,
        )

        self.ff = tfk.Sequential([
            tfk.layers.ReLU(),
            tfk.layers.Dense(ff_dims, kernel_initializer=kernel_initializer),
            tfk.layers.ReLU(),
            tfk.layers.Dense(attention_dims, kernel_initializer=kernel_initializer),
        ])

        if normalize:
            self.normalization_1 = tfk.layers.LayerNormalization(axis=-1)
            self.normalization_2 = tfk.layers.LayerNormalization(axis=-1)
        else:
            self.normalization_1 = None
            self.normalization_2 = None

    def call(self, qkv, attention_mask=None):
        _qkv = self.att(
            query=qkv, value=qkv, attention_mask=attention_mask, 
        )
        qkv = qkv + _qkv
        if self.normalization_1 is not None:
            qkv = self.normalization_1(qkv)

        qkv = self.ff(qkv) + qkv
        if self.normalization_2 is not None:
            qkv = self.normalization_2(qkv)

        if attention_mask is None:
            return qkv

        return qkv, attention_mask

class TransformerAssigner(tfk.models.Model):
    def __init__(self, attention_blocks, attention_dims, num_heads=8, ff_dims=None,  kernel_initializer='glorot_normal', normalize=True):
        super().__init__()

        if ff_dims is None:
            ff_dims = 2*attention_dims
        self.attention_dims = attention_dims

        self.blocks = []
        for i in range(attention_blocks):
            self.blocks.append(SelfAttentionBlock(attention_dims,  num_heads, ff_dims=ff_dims, kernel_initializer=kernel_initializer, normalize=normalize))

        self.input_layer = tfk.layers.Dense(attention_dims, kernel_initializer=kernel_initializer)
        self.output_layer = tfk.layers.Dense(3, kernel_initializer=kernel_initializer)

    def call(self, inputs):
        """
        inputs : (xypos, mask)
        """
        xy, mask, h = inputs
        attention_mask = tf.matmul(mask, mask, transpose_b=True)
        attention_mask = attention_mask - tf.linalg.diag(tf.linalg.diag_part(attention_mask)) #No self messages

        pdiff = xy[...,:,None,:] - xy[...,None,:,:]
        pdist = tf.reduce_sum(tf.square(pdiff), axis=-1) * attention_mask
        phi = angle_between(xy[...,:,None,:], xy[...,None,:,:], deg=False) * attention_mask
        embed()
        XX
        enc  = tf.concat((
            pdist,
            tf.sin(phi),
            tf.cos(phi),
        ), axis=-1)

        enc = self.input_layer(enc)

        hpred = enc
        for block in self.blocks:
            hpred, _ = block(hpred, attention_mask)

        hpred = self.output_layer(hpred)
        loss = block_l1_loss(hpred, h, mask)
        self.add_metric(loss, name='Block L1 Loss')
        #loss = mse_loss(hpred, h, mask)
        #self.add_metric(loss, name='MSE Loss')

        self.add_loss(loss)

        mean_abs = tf.reduce_sum(tf.abs(hpred)) / tf.reduce_sum(mask)
        self.add_metric(mean_abs, name="Mean Absolute Value")

        hpred =  tf.cast(tf.round(hpred), 'int32')
        correct = tf.reduce_all(h == hpred, axis=-1)
        correct = tf.where(tf.squeeze(mask, -1)==1., correct, False)
        correct = tf.reduce_sum(tf.cast(correct, 'float32')) / tf.reduce_sum(mask)
        self.add_metric(correct, "Accuracy")

        return hpred

def block_l1_loss(hpred, htrue, mask=None):
    htrue = tf.cast(htrue, 'float32')
    loss = tf.sqrt(tf.reduce_sum(tf.square(htrue - hpred), axis=-1))
    if mask is not None:
        loss = mask * loss[...,None]
    loss = tf.reduce_sum(loss) 
    if mask is not None:
        loss = loss / tf.reduce_sum(mask)
    return loss

def mse_loss(hpred, htrue, mask=None):
    htrue = tf.cast(htrue, 'float32')
    loss = tf.reduce_sum(tf.square(htrue - hpred), axis=-1)
    if mask is not None:
        loss = mask * loss[...,None]
    loss = tf.reduce_sum(loss) 
    if mask is not None:
        loss = loss / tf.reduce_sum(mask)
    return loss


def angle_between(vec1, vec2, deg=True, eps=1e-32):
    """
    This function computes the angle between vectors along the last dimension of the input arrays.
    This version is a numerically stable one based on arctan2 as described in this post:
     - https://scicomp.stackexchange.com/a/27769/39858
    Parameters
    ----------
    vec1 : array
        An arbitrarily batched arry of vectors
    vec2 : array
        An arbitrarily batched arry of vectors
    deg : bool (optional)
        Whether angles are returned in degrees or radians. The default is degrees (deg=True).
    Returns
    -------
    angles : array
        A vector of angles with the same leading dimensions of vec1 and vec2.
    """
    v1 = vec1 / (eps + tf.linalg.norm(vec1, axis=-1)[..., None])
    v2 = vec2 / (eps + tf.linalg.norm(vec2, axis=-1)[..., None])
    x1 = tf.linalg.norm(v1 - v2, axis=-1)
    x2 = tf.linalg.norm(v1 + v2, axis=-1)
    alpha = 2.0 * tf.atan2(x1, x2)
    if deg:
        return np.rad2deg(alpha)
    return alpha

