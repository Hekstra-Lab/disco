#!/usr/bin/env python

from disco import Ball,Detector
import pandas as pd
from matplotlib import pyplot as plt
import gemmi
from os.path import abspath
from argparse import ArgumentParser


def main():
    parser = ArgumentParser("Generate random laue patterns from pdb data")
    parser.add_argument("-n", default=100, type=int, help="Number of patterns to draw")
    parser = parser.parse_args()

    prefix = '/'.join(abspath(__file__).split('/')[:-1])
    filename = prefix + "/../data/pdb_data.csv.bz2"
    data = pd.read_csv(filename)

    detector_dist = 200. #mm
    size_x = 0.1 #pixel size in mm
    size_y = 0.1 #pixel size in mm
    beam_x, beam_y = 2048.,2048. #beam center in px

    detector = Detector.from_detector_dist(detector_dist, size_x, size_y, beam_x, beam_y)

    f = plt.figure()
    lmin,lmax = 1., 1.2 #<-- a typical wavelength range in Å
    for i in range(parser.n):
        datum = data.sample()
        try:
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
            s1 = ball.get_random_scattered_beam_wavevectors()
        except:
            continue
        x, y = detector.project(s1)
        plt.clf()
        plt.plot(x, y, 'k.')
        plt.plot(beam_x, beam_y, 'xr')
        plt.xlim(0., 2*beam_x)
        plt.ylim(0., 2*beam_y)
        plt.draw()
        plt.pause(0.1)

