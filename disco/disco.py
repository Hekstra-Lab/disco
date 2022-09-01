import reciprocalspaceship as rs
import numpy as np
import gemmi


class Detector:
    def __init__(self, dmat):
        """
        dmat : array
            This is a 3x3 matrix that describes the detector position in the lab frame. 
            The lab coordinates of a pixel position, {x, y} are given by dmat @ {x, y, 1}
        """
        self.dmat = dmat

    @classmethod
    def from_detector_dist(cls, distance, size_x, size_y, beam_x, beam_y):
        """
        A simple detector on the z-axis with no tilt.

        Parameters
        ----------
        distance : float
            Detector distance in mm
        size_x : float
            Pixel size on the fast (x) dim of the detector in mm
        size_y : float
            Pixel size on the fast (y) dim of the detector in mm
        beam_x : float
            Beam center x-coordinate in pixels
        beam_y : float
            Beam center y-coordinate in pixels
        """
        ori = [-beam_x * size_x, -beam_y * size_y, distance]
        dmat = np.array([
            [size_x, 0., 0.],
            [0, size_y,  0.],
            ori,
        ])
        return cls(dmat)

    def project(self, s1):
        """
        Project scattered beam vectors onto the detector.

        Parameters
        ----------
        s1 : array
            n x 3 array of scattered beam wavevectors with last axis of dimension 3
        Returns
        -------
        x,y : array
        """
        # Normalize
        #s1 = s1 / np.linalg.norm(s1, axis=-1)[:,None]
        norm = np.sqrt(
            s1[...,0]*s1[...,0] + 
            s1[...,1]*s1[...,1] + 
            s1[...,2]*s1[...,2]
        )
        s1 /= norm[...,None]
        xya = s1 @ np.linalg.inv(self.dmat) 
        x,y = xya[:,0] / xya[:,2], xya[:,1] / xya[:,2]
        return x,y

class Ball:
    def __init__(self, cell, spacegroup, dmin, lambda_min, lambda_max, s0=(0, 0., 1.)):
        """
        Parameters
        ----------
        cell : gemmi.UnitCell
        spacegroup : gemmi.SpaceGroup
        dmin : float
            Highest resolution refleciotn in Å
        lambda_min : float
            The minimum wavelength of the X-ray beam in Å
        lambda_max : float
            The maximum wavelength of the X-ray beam in Å
        s0 : array (optional)
            The possibly normalized direction of the incoming x-ray vector in the lab frame.
            This is going to default to +z (0, 0, 1)
        """
        self.cell = cell
        self.spacegroup = spacegroup
        self.dmin = dmin
        self.Hall = rs.utils.generate_reciprocal_cell(self.cell, self.dmin)
        self.Hall = self.Hall[~rs.utils.is_absent(self.Hall, self.spacegroup)]
        self.s0 = np.array(s0)
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.s0 = s0 / np.linalg.norm(s0)

    def s1_from_R(self, R, return_millers=False):
        """
        Get s1 wavevectors from rotation matrix, R
        """
        # See https://dials.github.io/documentation/conventions.html
        B = self.cell.fractionalization_matrix
        Qall = (R@B@self.Hall.T).T

        # My own calculation (sorry)
        wavelength = -2. * np.sum(Qall * self.s0[None,:], axis=-1) / np.sum(Qall*Qall, axis=-1)
        feasible = (wavelength >= self.lambda_min) & (wavelength <= self.lambda_max)
        s1 = Qall + self.s0[None,:]/wavelength[:,None]
        #s1 = Qall * wavelength[:, None] + self.s0
        if return_millers:
            return s1[feasible], self.Hall[feasible]
        return s1[feasible]

    def get_random_scattered_beam_wavevectors(self, return_millers=False):
        """
        Randomly generate a rotation matrix, and use it to compute all feasible scattered beam wavevectors.
        """
        from scipy.stats import special_ortho_group
        # Completely random unbiased rotation matrix in 3D
        R = special_ortho_group(3).rvs()
        return self.s1_from_R(R, return_millers)

