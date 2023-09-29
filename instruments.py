import numpy as np
from hcipy import *

def _mueller_rotate(angle):
    angle *= 2
    return np.array([[1,0,0,1],
                     [0,np.cos(angle),np.sin(angle),0],
                     [0,-np.sin(angle),np.cos(angle),0],
                     [0,0,0,1]])

def _stokes_list_to_array(stokes):

    # digest as numpy array
    stokes = np.asarray(stokes)
    stokes = np.moveaxis(stokes,0,-1) # stuff the stokes dimension in the back
    stokes = stokes[...,np.newaxis] # make it a vector

    return stokes

def _stokes_array_to_list(stokes):

    # digest as list
    stokes_list = []
    for i in range(4):
        stokes_list.append(stokes[...,i,0])
    return stokes_list

def mueller_rotation(M,angle):
    return _mueller_rotate(angle) @ M @ _mueller_rotate(-angle)

def polarizer_mueller(angle):

    # horizontal polarizer
    mueller = np.array([[1,1,0,0],
                        [1,1,0,0],
                        [0,0,0,0],
                        [0,0,0,0]])*0.5
    
    return mueller_rotation(mueller,angle)

def hwp_mueller(angle):
    mueller = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,-1,0],
                        [0,0,0,-1]])
    
    return mueller_rotation(mueller,angle)


def polarizing_beam_splitter(stokes):
    """propagate a stokes image through a PBS

    Parameters
    ----------
    stokes : list
        list of stokes images

    Returns
    -------
    list,list
        list of propagated stokes vectors for the horizontal, then vertical transmission
    """
    stokes = _stokes_list_to_array(stokes)

    # construct the polarizers
    hpol = polarizer_mueller(0)
    vpol = polarizer_mueller(np.pi/2)

    stokes_h = hpol @ stokes
    stokes_v = vpol @ stokes

    # turn back into a list
    h_channel = []
    v_channel = []
    for i in range(4):
        h_channel.append(stokes_h[...,i,0])
        v_channel.append(stokes_v[...,i,0])

    return h_channel,v_channel

def double_difference_stokes(stokes,normalize=False):

    pol_angles = [0,np.pi/4,np.pi/2,3*np.pi/4]
    hwp_angles = [i/2 for i in pol_angles]
    single_differences = []
    single_sums = []

    stokes_array = _stokes_list_to_array(stokes)

    for t in hwp_angles:

        hwp = hwp_mueller(t)
        stokes_rotate = hwp @ stokes_array
        stokes_rotate = _stokes_array_to_list(stokes_rotate)

        h,v = polarizing_beam_splitter(stokes_rotate)

        # record intensity data
        single_differences.append(h[0]-v[0])
        single_sums.append(h[0]+v[0])

    
    # single_differences now populated with Q, U, -Q, -U
    Q = (single_differences[0] - single_differences[2]) / 2
    U = (single_differences[1] - single_differences[3]) / 2

    if normalize:
        Q /= np.sum(single_sums,axis=0)
        U /= np.sum(single_sums,axis=0)

    return Q,U

def double_different_pdi(wf_star,wf_planet,prop,normalize=True,planet_contrast=1e-7):
    """propagates an HCIPy wavefront through a polarimeter that includes half-wave plate cycles
    to perform double-difference stokes polarimetry. Essentially a re-skin of the tutorial 
    Intro to Polarization in HCIPy.

    Parameters
    ----------
    wf_star : hcipy.Wavefront
        Wavefront in pupil plane with some input_stokes_parameter defined for the on-axis field
    wf_planet : hcipy.Wavefront
        Wavefront in pupil plane with some input_stokes_parameter defined for the exoplanet to measure
    prop : hcipy.FraunhoferPropagator
        propagator to prop from pupil grid to focal grid
    normalize : bool
        Whether to return the stokes parameters normalized or not. Defaults to true
    planet_contrast : float
        maximum power on the focused wf_planet field, defaults to 1e-7

    Returns
    -------
    numpy.ndarray,numpy.ndarray
        shaped arrays containing stokes Q and U
    """

    # initialize the polarimeter elements
    pbs_angles = [0,np.pi/4,np.pi/2,3*np.pi/4]
    single_differences = []
    single_sums = []
    wf_planet.electric_field *= np.sqrt(planet_contrast)
    for angle in pbs_angles:

        pbs = LinearPolarizingBeamSplitter(angle)

        img_star_1, img_star_2 = pbs(prop(wf_star))
        img_planet_1, img_planet_2 = pbs(prop(wf_planet))

        science_image_1 = img_star_1.power + img_planet_1.power
        science_image_2 = img_star_2.power + img_planet_2.power

        # adding noise
        # science_image_1 = large_poisson(science_image_1)
        # science_image_2 = large_poisson(science_image_2)
        diff = science_image_1 - science_image_2
        sums = science_image_1 + science_image_2
        single_differences.append(diff)
        single_sums.append(sums)

    # single_differences now populated with Q, U, -Q, -U
    Q = (single_differences[0] - single_differences[2]) / 2
    U = (single_differences[1] - single_differences[3]) / 2
    
    if normalize:
        Q /= np.sum(single_sums,axis=0)
        U /= np.sum(single_sums,axis=0)
    
    Q = Q.shaped
    U = U.shaped
    return Q,U