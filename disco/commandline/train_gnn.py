#!/usr/bin/env python

from IPython import embed
from disco import Ball
import pandas as pd
from matplotlib import pyplot as plt
import gemmi
from os.path import abspath
from argparse import ArgumentParser
from disco.layers import *
from disco.egnn import *

# Handle GPU selection
def set_gpu(gpu_id):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(gpus)
    if gpus:
        try:
            if gpu_id is None:
                tf.config.experimental.set_visible_devices([], 'GPU')
            else:
                tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)

gpu_id = 0
set_gpu(gpu_id)

prefix = '/'.join(abspath(__file__).split('/')[:-1])
filename = prefix + "/../data/pdb_data.csv.bz2"

min_reflections_per_image=32
max_reflections_per_image=256
batch_size=3
max_images = 10_000_000_000
epochs=10
steps_per_epoch=100

eager=False
#eager=True
layer_depth = 4
hidden_dim = 12
blocks = 6
dmodel = 16
df = pd.read_csv(filename)
lmin,lmax = 1., 1.02 #<-- a typical wavelength range in Å

kernel_initializer = tfk.initializers.VarianceScaling(scale=2., mode='fan_avg')
#kernel_initializer = 'he_normal'
activation="swish"


cell =[34., 45., 98., 90., 90., 90.]
spacegroup = 19
dmin = 3.

from reciprocalspaceship.decorators import cellify,spacegroupify

@cellify
@spacegroupify
def isomorphous_data_generator(cell, spacegroup, dmin, lmin, lmax, max_reflections_per_image, max_images, hmax=50):
    ball = Ball(cell, spacegroup, dmin, lmin, lmax)
    for i in range(max_images):
        s1,h = ball.get_random_scattered_beam_wavevectors(return_millers=True)
        s1_norm = np.sqrt(np.square(s1[...,0]) + np.square(s1[...,1]) + np.square(s1[...,2]))
        s1 = s1 / s1_norm[...,None] #Normalize
        q = s1 - ball.s0

        #Must remove millers outside hmax range
        idx = np.all(np.abs(h) < hmax, axis=-1)
        q = q[idx]
        h = h[idx]

        if len(h) < min_reflections_per_image:
            continue

        #Sample data to fit in model
        n = np.random.randint(min_reflections_per_image, max_reflections_per_image)
        n = min(n, len(h)) #Can't return more millers than there are
        idx = np.random.choice(np.arange(len(h)), size=n, replace=False)
        q = q[idx]
        h = h[idx]

        q = np.pad(q, [[0, max_reflections_per_image - n], [0, 0]])
        h = np.pad(h, [[0, max_reflections_per_image-n], [0, 0]])
        mask = np.concatenate((np.ones(n), np.zeros(max_reflections_per_image - n)))
        q, h, mask = tf.convert_to_tensor(q, dtype='float32'), tf.convert_to_tensor(h, dtype='int32'), tf.convert_to_tensor(mask, dtype='float32')
        mask = mask[None,...,None]
        q = q[None,...]
        h = h[None,...]
        yield (q, mask, h), h, mask


#generatorator = lambda: data_generator(df, max_reflections_per_image, max_images, hmax)
generatorator = lambda: isomorphous_data_generator(cell, spacegroup, dmin, lmin, lmax, max_reflections_per_image, max_images)

#edge_model = Resnet(blocks, dmodel)
#node_model = Resnet(blocks, dmodel, output_shape=1)
#hidden_model = Resnet(blocks, dmodel, output_shape=hidden_dim)

#edge_model = MLP(blocks)
#node_model = MLP(blocks, output_shape=1)
#hidden_model = MLP(blocks, output_shape=hidden_dim)

egnn = EGNN([EGCLayer.from_depth_hidden_dim(layer_depth, hidden_dim, activation=activation) for i in range(blocks)])
#egnn = EGNN([EGCLayer.from_depth_hidden_dim(5, 16, steps=5)])

assigner = Assigner(egnn)

inputs = next(generatorator())
sig = tf.nest.map_structure(tf.TensorSpec.from_tensor, inputs)
data = tf.data.Dataset.from_generator(generatorator, output_signature=sig)
data = data.unbatch().batch(batch_size)


opt = tfk.optimizers.Adam()
assigner.compile(opt, run_eagerly=eager)

history = assigner.fit(data, epochs=epochs, steps_per_epoch=steps_per_epoch)

embed(colors='linux')
