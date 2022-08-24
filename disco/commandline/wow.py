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
    parser.add_argument("-s", default=None, type=str, help='Save movie of your `wow` session to an mp4 file')
    parser.add_argument("--csv", default=None, type=str, help="Save the coordinates of spots to a csv file.")
    parser = parser.parse_args()

    prefix = '/'.join(abspath(__file__).split('/')[:-1])
    filename = prefix + "/../data/pdb_data.csv.bz2"
    data = pd.read_csv(filename)

    detector_dist = 200. #mm
    size_x = 0.1 #pixel size in mm
    size_y = 0.1 #pixel size in mm
    beam_x, beam_y = 2048.,2048. #beam center in px

    detector = Detector.from_detector_dist(detector_dist, size_x, size_y, beam_x, beam_y)

    f = plt.figure(figsize=(10, 10))
    from celluloid import Camera
    cam = Camera(f)
    lmin,lmax = 1., 1.2 #<-- a typical wavelength range in Å
    dfout = []
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
        plt.plot(x, y, 'k.', alpha=0.3)
        plt.plot(beam_x, beam_y, 'xr')
        plt.xlim(0., 2*beam_x)
        plt.ylim(0., 2*beam_y)
        cam.snap()
        if parser.csv is not None:
            df = pd.DataFrame({
                'x' : x,
                'y' : y,
            })
            df['frame'] = i
            dfout.append(df)

    anim = cam.animate()
    if parser.s is not None:
        if parser.s.endswith(".gif"):
            anim.save(parser.s, writer='imagemagick')
            print("HIHIHII")
        elif parser.s.endswith(".mp4"):
            anim.save(parser.s)
    if parser.csv is not None:
        dfout = pd.concat(dfout)
        dfout.to_csv(parser.csv)
    plt.show()
