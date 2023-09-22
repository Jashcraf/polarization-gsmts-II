import numpy as np
from hcipy import *

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