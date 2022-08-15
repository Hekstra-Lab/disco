#!/usr/bin/env python

from IPython import embed
from disco import Ball
import pandas as pd
from matplotlib import pyplot as plt
import gemmi
from os.path import abspath
from argparse import ArgumentParser
from disco.transformer import *


prefix = '/'.join(abspath(__file__).split('/')[:-1])
filename = prefix + "/../data/pdb_data.csv.bz2"

min_reflections_per_image=511
max_reflections_per_image=512
batch_size=10
max_images = 10_000_000_000
hmax = 50
epochs=50
steps_per_epoch=100
normalize=True

eager=False
eager=True

blocks = 5
dmodel = 256

df = pd.read_csv(filename)
lmin,lmax = 1., 1.2 #<-- a typical wavelength range in Å

#kernel_initializer = tfk.initializers.VarianceScaling(scale=1/5./blocks, mode='fan_avg')
kernel_initializer = 'glorot_normal'


cell =[34., 45., 98., 90., 90., 90.]
spacegroup = 19
dmin = 2.

from reciprocalspaceship.decorators import cellify,spacegroupify

@cellify
@spacegroupify
def isomorphous_data_generator(cell, spacegroup, dmin, lmin, lmax, max_reflections_per_image, max_images, hmax=50):
    ball = Ball(cell, spacegroup, dmin, lmin, lmax)
    for i in range(max_images):
        s1,h = ball.get_random_scattered_beam_wavevectors(return_millers=True)
        s1_norm = np.sqrt(np.square(s1[...,0]) + np.square(s1[...,1]) + np.square(s1[...,2]))
        s1 = s1 / s1_norm[...,None]
        s1 = s1 / s1[...,2,None]

        #These are all scalled to have the same z component
        #TODO: add random detector tilts
        xypos = s1[...,:2]

        #Must remove millers outside hmax range
        idx = np.all(np.abs(h) < hmax, axis=-1)
        xypos = xypos[idx]
        h = h[idx]

        if len(h) < min_reflections_per_image:
            continue

        #Sample data to fit in model
        n = np.random.randint(min_reflections_per_image, max_reflections_per_image)
        n = min(n, len(h)) #Can't return more millers than there are
        idx = np.random.choice(np.arange(len(h)), size=n, replace=False)
        xypos = xypos[idx]
        h = h[idx]

        pdist = np.square(xypos[...,None,:] - xypos[...,:None]).sum(-1)
        #xypos = np.vstack([np.histogram(i, np.linspace(0., 1., dmodel))[0] for i in pdist])


        xypos = np.pad(xypos, [[0, max_reflections_per_image - n], [0, 0]])
        h = np.pad(h, [[0, max_reflections_per_image-n], [0, 0]])
        mask = np.concatenate((np.ones(n), np.zeros(max_reflections_per_image - n)))
        xypos, h, mask = tf.convert_to_tensor(xypos, dtype='float32'), tf.convert_to_tensor(h, dtype='int32'), tf.convert_to_tensor(mask, dtype='float32')
        mask = mask[None,...,None]
        xypos = xypos[None,...]
        h = h[None,...]
        yield (xypos, mask, h), h, mask

#generatorator = lambda: data_generator(df, max_reflections_per_image, max_images, hmax)
generatorator = lambda: isomorphous_data_generator(cell, spacegroup, dmin, lmin, lmax, max_reflections_per_image, max_images, hmax)

inputs = next(generatorator())
sig = tf.nest.map_structure(tf.TensorSpec.from_tensor, inputs)
data = tf.data.Dataset.from_generator(generatorator, output_signature=sig)
data = data.unbatch().batch(batch_size)

metrics = [
    #'categorical_accuracy'
]

#assigner = Assigner(blocks, dmodel, hmax=hmax, kernel_initializer=kernel_initializer, normalize=normalize)
assigner = TransformerAssigner(blocks, dmodel, kernel_initializer=kernel_initializer, normalize=normalize)
opt = tfk.optimizers.Adam(1e-3)
assigner.compile(opt, run_eagerly=eager)

history = assigner.fit(data, epochs=epochs, steps_per_epoch=steps_per_epoch)

embed(colors='linux')
