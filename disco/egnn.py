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

class Resnet(tfk.layers.Layer):
    def __init__(self, num_blocks, units, output_shape=None, kernel_initializer='glorot_normal', **kwargs):
        super().__init__()
        self.layers = []
        for i in range(num_blocks):
            self.layers.append(ResNetLayer(units, kernel_initializer=kernel_initializer, **kwargs))

        if output_shape is not None:
            self.output_layer = tfk.layers.Dense(output_shape, kernel_initializer=kernel_initializer)
        else:
            self.output_layer = None

    def call(self, x, mask=None):
        out = x
        for layer in self.layers:
            out = layer(out, mask=mask)

        if self.output_layer is not None:
            out = self.output_layer(out)
        return out

class MLPLayer(tfk.layers.Layer):
    """ A lazy layer that takes it shape from the input at compile time """
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def build(self, shape, **kwargs):
        units = shape[-1]
        self.dense = tfk.layers.Dense(units, **self.kwargs)

    def call(self, x, **kwargs):
        return self.dense(x)

class MLP(tfk.layers.Layer):
    def __init__(self, num_blocks, output_shape=None, activation='ReLU', kernel_initializer='glorot_normal', **kwargs):
        super().__init__()
        self.layers = []

        for i in range(num_blocks):
            self.layers.append(MLPLayer(kernel_initializer=kernel_initializer, activation=activation))

        if output_shape is not None:
            self.output_layer = tfk.layers.Dense(output_shape, kernel_initializer=kernel_initializer)
        else:
            self.output_layer = None

    def call(self, x, mask=None):
        out = x
        for layer in self.layers:
            out = layer(out)

        if self.output_layer is not None:
            out = self.output_layer(out)
        return out

class EGCLayer(tfk.layers.Layer):
    def __init__(self, edge_model, node_model, hidden_model, steps=1, kernel_initializer='glorot_normal', **kwargs):
        """
        edge_model : callable
            A callable that maps from R^{nf} -> R^{nf}
        node_model : callable
            A callable that maps from R^{nf} -> R^1
        hidden_model : callable
            A callable that maps from R^{2nf} -> R^{nf}
        """
        super().__init__()
        self.hidden_model = hidden_model
        self.edge_model = edge_model
        self.node_model = node_model
        self.steps = steps

    @classmethod
    def from_depth_hidden_dim(cls, depth, hidden_dim, kernel_initializer='glorot_normal', activation='gelu', **kwargs):
        edge_model = MLP(depth, kernel_initializer=kernel_initializer, activation=activation)
        node_model = MLP(depth, output_shape=1, kernel_initializer=kernel_initializer, activation=activation)
        hidden_model = MLP(depth, output_shape=hidden_dim, kernel_initializer=kernel_initializer, activation=activation)
        return cls(edge_model, node_model, hidden_model, **kwargs)

    def call(self, nodes, hidden, mask=None):
        if mask is None:
            mask = np.ones_like(nodes[...,0])
        mask = tf.cast(mask, 'float32')

        mask_2d = tf.matmul(mask, mask, transpose_b=True)
        mask_2d = mask_2d - tf.linalg.diag(tf.linalg.diag_part(mask_2d)) #No self messages

        for i in range(self.steps):
            dnode = (nodes[...,:,None,:] - nodes[...,None,:,:]) * mask_2d[...,None]
            pdist = tf.reduce_sum(tf.square(dnode), axis=-1, keepdims=True)

            message_input = tf.concat((
                tf.ones_like(pdist)*hidden[...,None,:],
                tf.ones_like(pdist)*hidden[...,None,:,:],
                pdist,
            ), axis=-1)  

            messages = self.edge_model(message_input)
            weights = self.node_model(messages) 
            C = tf.reduce_sum(mask_2d, axis=-1)
            C = tf.where(C==0., 0., tf.math.reciprocal(C))
            weights = C[...,None,None] * weights * mask_2d[...,None]
            nodes = nodes + tf.reduce_sum(dnode * weights, axis=-2, keepdims=False)

            hidden_input = tf.concat((
                hidden,
                tf.reduce_sum(messages, axis=-2),
            ), axis=-1)
            hidden = self.hidden_model(hidden_input)

        return nodes, hidden

class EGNN(tfk.models.Model):
    def __init__(self, egclayers, kernel_initializer='glorot_normal', **kwargs):
        """
        edge_model : callable
            A callable that maps from R^{nf} -> R^{nf}
        node_model : callable
            A callable that maps from R^{nf} -> R^1
        hidden_model : callable
            A callable that maps from R^{2nf} -> R^{nf}
        """
        super().__init__()
        self.egclayers = egclayers
        self.hidden_dims = self.egclayers[0].hidden_model.output_layer.units

    def call(self, nodes, mask=None):
        hidden = tf.ones_like(nodes[...,0,None]) * tf.ones(self.hidden_dims)[...,:]
        for layer in self.egclayers:
            nodes, hidden = layer(nodes, hidden, mask)
        return nodes

class Assigner(tfk.models.Model):
    def __init__(self, gnn, **kwargs):
        super().__init__(**kwargs)
        self.gnn = gnn

    def call(self, inputs, **kwargs):
        xyz, mask, h = inputs
        hpred = self.gnn(xyz, mask)
        #loss = mse_loss(hpred, h, mask)
        #self.add_metric(loss, name='MSE Loss')

        loss = block_l1_loss(hpred, h, mask)
        self.add_metric(loss, name='Block L1 Loss')

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

