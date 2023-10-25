"""Contains GSMT forward modeling situation"""

from hcipy import *
from poke.poke_core import Rayfront
from poke.interfaces import jones_pupil_to_hcipy_wavefront
from matplotlib import pyplot as plt
import numpy as np
# from filter_specs import *
import os
from prysm import (
    coordinates,
    geometry,
    segmented,
    polynomials,
    propagation,
    wavelengths,
)
from prysm.conf import config

# input should be like 
sim_params = {
    'wavelength' : 0.806,
    'order' : 6,
    'npix_pupil' : 1024,
    'segment variation' : True,
	'q_focal' : 5,
	'nairy_focal' : 15
}

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

gmt_conf = {
	'GSMT' : 'GMT',
    'D_tel' : 25.448,
    'F_tel' : 207.589,
    'pth' : 'C:/Users/UASAL-OPTICS/Desktop/polarization-gsmts-II/raytraces/GMT.zmx',
	'aperture' : make_gmt_aperture(normalized=False,with_spiders=False),
	'aberration ptv' : 0.02, # fraction of nominal
	'nominal thickness' : 4.12e-9, # nm
	'n' : -2, # aberration exponent
	'coating index' : 1.6708,
	'substrate index' : 2.7671 + 1j*8.3172
}

elt_conf = {
	'GSMT' : 'ELT',
    'D_tel' : 39.1464,
    'F_tel' : 646.794,
    'pth' : 'C:/Users/UASAL-OPTICS/Desktop/polarization-gsmts-II/raytraces/ELT.zmx',
	'aberration ptv' : 0.10, # fraction of nominal
	'nominal thickness' : 5.5e-9, # nm
	'coating index' : 2.0249, # SiN
	'substrate index' : 0.077492 + 1j*5.3462 # Ag
}

tmt_conf = {
	'GSMT' : 'TMT',
    'D_tel' : 30,
    'F_tel' : 449.99,
    'pth' : 'C:/Users/UASAL-OPTICS/Desktop/polarization-gsmts-II/raytraces/TMT.zmx',
	'aberration ptv' : 0.10, # fraction of nominal
	'nominal thickness' : 8.5e-9, # nm
	'coating index' : 2.0249, # SiN
	'substrate index' : 0.077492 + 1j*5.3462 # Ag
}


def create_tmt_aperture_and_phase(diameter=tmt_conf['D_tel'],
								ref_thickness=tmt_conf['nominal thickness'],
								pv_frac=tmt_conf['aberration ptv'],
								npix=sim_params['npix_pupil'],
								nmodes=4):
	"""uses prysm to generate the tmt aperture with a mode basis"""

	# update the surface with segment thickness, maybe a mask_rays() function?
	# assumption of rotational symmetry

	# compose segmented
	x, y = coordinates.make_xy_grid(npix, diameter=diameter)
	dx = x[0,1] - x[0,0]

	rings = 6
	flat_to_flat_to_vertex_vertex = 2 / np.sqrt(3)
	vtov_to_flat_to_flat = 1 / flat_to_flat_to_vertex_vertex

	segdiam = vtov_to_flat_to_flat * 1.44
	exclude = [
		0, 1, 2, 3, 4, 5, 6, # center
		469, 470, 508, 509, 507, 510, 506, 545,
		471, 511, 505, 544, 472, 397, 433, 546, # top, bottom
		534, 533, 532, 531, 521, 522, 523, 524, # left edge
		482, 483, 484, 485, 495, 494, 493, 492, # right edge
		457, 535, 445, 520, 481, 409, 421, 496, # corners
		536, 537, 479, 480, 497, 498, 519, 518, # next 'diagonal' from corners
	]
	cha = segmented.CompositeHexagonalAperture(x,y,13,segdiam,0.0025,exclude=exclude)
	nms = [polynomials.noll_to_nm(j) for j in range(nmodes+1)]
	cha.prepare_opd_bases(polynomials.zernike_nm_sequence, nms, normalization_radius=1.44/2);
	basis_coefs = np.zeros((len(cha.segment_ids), len(nms)), dtype=config.precision)
	basis_coefs[:, 0] = np.random.uniform(-500, 500, 492)
	basis_coefs[:, 1] = np.random.uniform(-1000, 1000, 492)
	basis_coefs[:, 2] = np.random.uniform(-1000, 1000, 492)
	basis_coefs[:, 3:] = np.random.uniform(-1000, 1000, 492*(nmodes-2)).reshape((492,nmodes-2))
	phase_map = cha.compose_opd(basis_coefs)
	phase_map /= np.max(np.abs(phase_map))
	phase_map *= ref_thickness*pv_frac
	phase_map += ref_thickness

	return cha.amp.ravel(),phase_map.ravel()

tmt_conf.update({'aperture':create_tmt_aperture_and_phase()})


def create_elt_aperture_and_phase(diameter=elt_conf['D_tel'],
								ref_thickness=elt_conf['nominal thickness'],
								pv_frac=elt_conf['aberration ptv'],
								npix=sim_params['npix_pupil'],
								nmodes=4,
								display_numbered_pupils=False):
	
	# source: https://elt.eso.org/mirror/M1/
	
	
	# compose segmented
	x, y = coordinates.make_xy_grid(npix, diameter=diameter)
	dx = x[0,1] - x[0,0]

	rings = 6
	flat_to_flat_to_vertex_vertex = 2 / np.sqrt(3)
	vtov_to_flat_to_flat = 1 / flat_to_flat_to_vertex_vertex

	segdiam = vtov_to_flat_to_flat * 1.45

	# The secret is literally shaving off the corners until you see a flat of 5 segments
	exclude = [
		817, 918, 818, 819, 721, 917, 820, 722, 816, 916, # top
		868, 867, 869, 866, 769, 870, 865, 768, 770, 871, # bottom
		902, 901, 903, 900, 801, 904, 899, 800, 802, 905, # top left
		834, 835, 833, 836, 737, 832, 837, 738, 736, 831, # top right
		885, 884, 886, 883, 785, 887, 882, 784, 786, 888, # bottom left
		851, 852, 850, 853, 753, 849, 854, 754, 752, 848 # bottom right
	]

	# the centers are easier to yeet by appending, but you can type them out yourself if you wish
	for i in range(61):
		exclude.append(i)
	
	cha = segmented.CompositeHexagonalAperture(x,y,17,segdiam,0.004, exclude=exclude)

	if display_numbered_pupils:
		fig, ax = plt.subplots(figsize=(15,15))
		ax.imshow(cha.amp, origin='lower', cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()])
		for center, id_ in zip(cha.all_centers, cha.segment_ids):
			plt.text(*center, id_, ha='center', va='center')
		plt.show()

	nms = [polynomials.noll_to_nm(j) for j in range(nmodes+1)]
	cha.prepare_opd_bases(polynomials.zernike_nm_sequence, nms, normalization_radius=1.45/2)
	basis_coefs = np.zeros((len(cha.segment_ids), len(nms)), dtype=config.precision)
	basis_coefs[:, 0] = np.random.uniform(-500, 500, 798)
	basis_coefs[:, 1] = np.random.uniform(-1000, 1000, 798)
	basis_coefs[:, 2] = np.random.uniform(-1000, 1000, 798)
	basis_coefs[:, 3:] = np.random.uniform(-1000, 1000, 798*(nmodes-2)).reshape((798,nmodes-2))
	phase_map = cha.compose_opd(basis_coefs)
	phase_map /= np.max(np.abs(phase_map))
	phase_map *= ref_thickness*pv_frac
	phase_map += ref_thickness

	return cha.amp.ravel(),phase_map.ravel()


elt_conf.update({'aperture':create_elt_aperture_and_phase()})
	

def sim_gsmt_jones_pupil(sim_params, tele_conf):

	# set up the problem
	D_tel = tele_conf['D_tel']
	F_tel = tele_conf['F_tel']
	npix = sim_params['npix_pupil']
	wvl = sim_params['wavelength']
	coating_index = tele_conf['coating index']
	substrate_index = tele_conf['substrate index'] 

	# Assemble grids
	grid = make_pupil_grid(npix,D_tel)
	focal_grid = make_focal_grid(sim_params['q_focal'], sim_params['nairy_focal'], spatial_resolution=wvl / D_tel * F_tel)

	# create propagator
	prop = FraunhoferPropagator(grid, focal_grid, focal_length=F_tel)


	if tele_conf['GSMT'] == 'GMT':
		
		# construct aperture
		aperture = evaluate_supersampled(tele_conf['aperture'],grid,1)
		
		# create the surface aberration
		if sim_params['segment variation'] == True:
			exponent = tele_conf['n']
			nominal_ptv = 0.02 * tele_conf['nominal thickness']
			layer = SurfaceAberration(grid, ptv=nominal_ptv, diameter=np.sqrt(2)*D_tel, exponent=exponent).phase(wvl)
			layer /= np.max(layer)
			layer *= nominal_ptv
			layer += tele_conf['nominal thickness'] # a raveled field
		else:
			layer = tele_conf['nominal thickness']

	else:

		# construct aperture
		aperture,phase = tele_conf['aperture']
		aperture = Field(aperture,grid)

		# create the low-order aberrations
		if sim_params['segment variation'] == True:
			layer = phase

		else:
			layer = np.full_like(phase,tele_conf['nominal thickness'])

	# Multilayer coating for Poke
	variable_layer = [
		(np.full_like(layer,coating_index,dtype=np.complex128),np.copy(layer)),
		(np.full_like(layer,substrate_index,dtype=np.complex128))
	]

	nominal_layer = [
		(np.full_like(layer,coating_index,dtype=np.complex128),np.full_like(layer,tele_conf['nominal thickness'])),
		(np.full_like(layer,substrate_index,dtype=np.complex128))
	]

	# do the PRT
	if tele_conf['GSMT'] == 'ELT':
		# do five surfaces

		m1 = {'surf':1,'coating':variable_layer,'mode':'reflect'}
		m2 = {'surf':2,'coating':nominal_layer,'mode':'reflect'}
		m3 = {'surf':3,'coating':nominal_layer,'mode':'reflect'}
		m4 = {'surf':5,'coating':nominal_layer,'mode':'reflect'}
		m5 = {'surf':9,'coating':nominal_layer,'mode':'reflect'}
		surflist = [m1,m2,m3,m4,m5]

	else:
		# it's a 3 surface system
		m1 = {'surf':1,'coating':variable_layer,'mode':'reflect'}
		m2 = {'surf':2,'coating':nominal_layer,'mode':'reflect'}
		m3 = {'surf':5,'coating':nominal_layer,'mode':'reflect'}
		surflist = [m1,m2,m3]
	
	# Poke is yummy
	rf = Rayfront(npix,wvl,D_tel/2,max_fov=1e-3,circle=False)
	rf.as_polarized(surflist)
	rf.trace_rayset(tele_conf['pth'])
	rf.wavelength = wvl

	rf.compute_jones_pupil(aloc=np.array([0.,1.,0.]))

	wvfnt = jones_pupil_to_hcipy_wavefront(rf.jones_pupil,grid,shape=npix)
	wvfnt.electric_field *= aperture
	wvfnt.total_power = 1.0
	wvfnt.wavelength = wvl

	# apply perfect AO system
	avg_phase = (wvfnt.phase[0,0] + wvfnt.phase[1,1]) / 2
	wvfnt.electric_field *= np.exp(-1j*avg_phase)

	norm = prop(wvfnt).power.max()
	coronagraph = PerfectCoronagraph(aperture, sim_params['order'])
	wfout = prop(coronagraph(wvfnt))

	return wfout.power / norm