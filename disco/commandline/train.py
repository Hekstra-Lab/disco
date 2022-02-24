#!/usr/bin/env python

from IPython import embed
from disco import Ball
import pandas as pd
from matplotlib import pyplot as plt
import gemmi
from os.path import abspath
from argparse import ArgumentParser
from disco.layers import *


prefix = '/'.join(abspath(__file__).split('/')[:-1])
filename = prefix + "/../data/pdb_data.csv.bz2"

max_reflections_per_image=512 
min_reflections_per_image=20
batch_size=1
max_images = 10_000
hmax = 50
epochs=1
label_smoothing=0.1

df = pd.read_csv(filename)
lmin,lmax = 1., 1.2 #<-- a typical wavelength range in Å


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
        xypos = np.pad(xypos, [[0, max_reflections_per_image - n], [0, 0]])
        h = h + hmax #No negative values
        h = np.pad(h, [[0, max_reflections_per_image-n], [0, 0]])
        mask = np.concatenate((np.ones(n), np.zeros(max_reflections_per_image - n)))
        xypos, h, mask = tf.convert_to_tensor(xypos, dtype='float32'), tf.convert_to_tensor(h, dtype='int32'), tf.convert_to_tensor(mask, dtype='float32')
        h = tf.one_hot(h, 2*hmax+1)
        mask = mask[None,...,None]
        xypos = xypos[None,...]
        h = h[None,...]
        yield (xypos, mask), h, mask


"""
def data_generator(data_frame, max_reflections_per_image, max_images, hmax=50):
    for i in range(max_images):
        try:
            datum = data_frame.sample()
            cell = gemmi.UnitCell(
                datum['Length a (Å)'],
                datum['Length b (Å)'],
                datum['Length c (Å)'],
                datum['Angle alpha (°)'],
                datum['Angle beta (°)'],
                datum['Angle gamma (°)'],
            )
            sg = gemmi.SpaceGroup(str(datum['Space Group'].iloc[0]))
            dmin = datum['High Resolution Limit'].to_numpy('float32')
            ball = Ball(cell, sg, dmin, lmin, lmax)
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
            xypos = np.pad(xypos, [[0, max_reflections_per_image - n], [0, 0]])
            h = h + hmax #No negative values
            h = np.pad(h, [[0, max_reflections_per_image-n], [0, 0]])
            mask = np.concatenate((np.ones(n), np.zeros(max_reflections_per_image - n)))
            xypos, h, mask = tf.convert_to_tensor(xypos, dtype='float32'), tf.convert_to_tensor(h, dtype='int32'), tf.convert_to_tensor(mask, dtype='float32')
            h = tf.one_hot(h, 2*hmax+1)
            mask = mask[None,...,None]
            xypos = xypos[None,...]
            h = h[None,...]
            yield (xypos, mask), (h, mask)
        except:
            continue
"""

#generatorator = lambda: data_generator(df, max_reflections_per_image, max_images, hmax)
generatorator = lambda: isomorphous_data_generator(cell, spacegroup, dmin, lmin, lmax, max_reflections_per_image, max_images, hmax)

inputs = next(generatorator())
sig = tf.nest.map_structure(tf.TensorSpec.from_tensor, inputs)
data = tf.data.Dataset.from_generator(generatorator, output_signature=sig)
data = data.unbatch().batch(batch_size)

blocks = 8
attention_dims = 16
num_heads = 24
ff_dims = attention_dims

assigner = Assigner(blocks, attention_dims, num_heads, ff_dims)
opt = tfk.optimizers.Adam()
loss = tfk.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
assigner.compile(opt, loss=loss, metrics='categorical_accuracy')

assigner.fit(data, epochs=epochs)

embed(colors='linux')
