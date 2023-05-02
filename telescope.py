from hcipy import *
import numpy as np

class Instrument:

    def __init__(self,wavelength,aperture,coronagraph,pupil_sampling,pupil_pixels,image_sampling,image_pixels):

        self.wavelength = wavelength
        self.aperture = aperture
        self.coronagraph = coronagraph
        self.pupil_sampling = pupil_sampling
        self.pupil_pixels = pupil_pixels
        self.image_sampling = image_sampling
        self.image_pixels = image_pixels


    def forward(self):

        # compute HCIPy PSF
        
        # Configure grids
        pupil_grid = make_pupil_grid(self.pupil_pixels,self.pupil_pixels*self.pupil_sampling)
        
        # image_grid = make_focal_grid()
        
        pass

    @property
    def wavelength(self):
        return self.wavelength
    
    @wavelength.setter
    def wavelength(self,wvl):
        self.wavelength = wvl

    @property
    def aperture(self):
        return self.aperture
    
    @aperture.setter
    def aperture(self,aperture):
        self.aperture = aperture

    @property
    def coronagraph(self):
        return self.coronagraph
    
    @coronagraph.setter
    def coronagraph(self,coron):
        self.coronagraph = coron

    @property
    def pupil_sampling(self):
        return self.pupil_sampling
    
    @pupil_sampling.setter
    def pupil_sampling(self,px):
        self.pupil_sampling = px

    @property
    def pupil_pixels(self):
        return self.pupil_pixels
    
    @pupil_pixels.setter
    def pupil_pixels(self,npix):
        self.pupil_pixels = npix

    @property
    def image_sampling(self):
        return self.image_sampling
    
    @image_sampling.setter
    def image_sampling(self,dx):
        self.image_sampling = dx

    @property
    def image_pixels(self):
        return self.image_pixels
    
    @image_pixels.setter
    def image_pixels(self,npix):
        self.image_pixels = npix

    
    

