"""Contains data about the GSMTs"""

from hcipy import *
from matplotlib import pyplot as plt
import numpy as np
from filter_specs import *
import os

filters = {
	'units':'microns',
	'U' : 0.364,
	'B' : 0.442,
	'g' : 0.454,
	'V' : 0.551,
	'R' : 0.658,
	'I' : 0.806,
	'z' : 0.900,
	'y' : 1.02,
	'J' : 1.24,
	'H' : 1.63
}

GMT = {
	'aperture': make_gmt_aperture,
	'diameter':
}

def load_jones_pupil(grid, filter, sub_path, coating):
	path = sub_path + '/' + coating + '/'
	
	jones_pupil = grid.zeros((2,2), dtype=complex)
	for i, elem in enumerate(['xx', 'xy', 'yx', 'yy']):
		file_real = 'E{:s}r_{:s}.fits'.format(elem, filter)
		file_imag = 'E{:s}i_{:s}.fits'.format(elem, filter)	
		data = Field( (read_fits(path + file_real) + 1j * read_fits(path + file_imag)).ravel(), grid)
		
		xi, yi = np.unravel_index(i, (2, 2))
		jones_pupil[yi, xi] = np.nan_to_num(data)
	
	return jones_pupil