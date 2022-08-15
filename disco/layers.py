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

class Pooling(tfk.layers.Layer):
    def __init__(self, normalize=False, **kwargs):
        super().__init__()
        self.normalize = normalize
        self.dense_kwargs = kwargs

    def build(self, shape, **kwargs):
        self.dense = tfk.layers.Dense(shape[-1] + 1, **self.dense_kwargs)
        self.softmax = tfk.layers.Softmax(axis=-1)
        if self.normalize:
            self.normalization = tfk.layers.LayerNormalization(axis=-2)
        self.out = tfk.layers.Dense(shape[-1], **self.dense_kwargs)

    def call(self, x, mask=None, **kwargs):
        out = x
        if self.normalize:
            out = self.normalization(out)
        out = self.dense(out)
        w,out = out[...,0],out[...,1:]

        w = self.softmax(w, mask=tf.squeeze(mask, axis=-1))
        out = tf.matmul(w[...,None,:], out)
        out = tf.concat((out * tf.ones_like(x), x), axis=-1)
        out = self.out(out)

        return out 

class MaskedSequential(tfk.models.Sequential):
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)

    def call(self, x, mask=None, **kwargs):
        out = x
        for layer in self.layers:
            out = layer(out, mask=mask)
        return out

class Assigner(tfk.models.Model):
    def __init__(self, num_blocks, dmodel,  ff_dims=None,  hmax=50, kernel_initializer='glorot_normal', normalize=True):
        super().__init__()
        if ff_dims is None:
            ff_dims = 2*dmodel
        self.ff_dims = ff_dims
        self.embed = tfk.layers.Dense(dmodel, kernel_initializer=kernel_initializer)
        self.dmodel = dmodel

        encoder_layers = []
        for i in range(num_blocks):
            encoder_layers += [
                ResNetLayer(ff_dims, kernel_initializer=kernel_initializer, normalize=normalize),
                Pooling(kernel_initializer=kernel_initializer, normalize=normalize),
            ]
        self.encoder = MaskedSequential(encoder_layers)

        #decoder_layers = []
        #decoder_layers.append(tfk.layers.Dense(3 * (2*hmax + 1), kernel_initializer='glorot_normal'))
        #decoder_layers.append(tfk.layers.Reshape((-1, 3, 2*hmax+1)))
        #decoder_layers.append(tfk.layers.Softmax(axis=-1))
        #self.decoder = tfk.models.Sequential(decoder_layers)

        #self.decoder = HKLDigitizer(hmax, kernel_initializer=kernel_initializer)
        self.decoder = tfk.layers.Dense(3, kernel_initializer='glorot_normal')

    def call(self, inputs):
        """
        inputs : (xypos, mask)
        """
        xy, mask, h = inputs
        pdiff = xy[...,:,None,:] - xy[...,None,:,:]
        pdist = tf.reduce_sum(tf.square(pdiff), axis=-1)
        phi = angle_between(xy[...,:,None,:], xy[...,None,:,:], deg=False)
        enc  = tf.concat((
            pdist,
            tf.sin(phi),
            tf.cos(phi),
        ), axis=-1)

        out = self.embed(enc)
        out = self.encoder(out, mask)
        out = self.decoder(out)

        htrue = tf.cast(h, 'float32')
        loss = tf.sqrt(tf.reduce_sum(tf.square(htrue - out), axis=-1))
        loss = tf.reduce_sum(mask * loss[...,None]) / tf.reduce_sum(mask)

        self.add_metric(loss, name='Block L1 Loss')
        self.add_loss(loss)

        mean_abs = tf.reduce_sum(tf.abs(out)) / tf.reduce_sum(mask)
        self.add_metric(mean_abs, name="Mean Absolute Value")

        hpred =  tf.round(out)
        correct = tf.reduce_all(htrue == hpred, axis=-1)
        correct = tf.where(tf.squeeze(mask, -1)==1., correct, False)
        correct = tf.reduce_sum(tf.cast(correct, 'float32')) / tf.reduce_sum(mask)
        self.add_metric(correct, "Accuracy")

        return out




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

