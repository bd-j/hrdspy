import numpy as np
import imf
import starmodel
import isochrone
import observate
import matplotlib.pyplot as pl

class Cluster(object):
    def __init__(self, target_mass, logage, Z, isoc = None, speclib = None, IMF = None,):
        self.target_mass = target_mass
        self.logage = logage
        self. Z = Z
        self.compose(isoc, speclib, IMF)
        
    def compose(self, isoc, speclib, IMF):
        if IMF is None:
            self.imf  = imf.SalpeterIMF()
        else:
            self.imf = IMF
        if isoc is None:
            self.isoc=isochrone.Padova2007()
            self.isoc.load_all_isoc()
        else:
            self.isoc = isoc
        if speclib is None:
            self.speclib = starmodel.BaSeL3()
            self.speclib.read_all_Z()
        else:
            self.speclib = speclib

        self.stars = starmodel.SpecLibrary()
        #self.stars.wavelength = self.speclib.wavelength
        
    def generate_stars(self):
        """Generate a population of stars from the IMF given a preset total mass for the cluster.
        Then, determine the parameters of these stars from interpolation of the isochrone values
        given the age and metallicity of the population.  Returns a structured array of shape (nstar)
        where each field of the structure is a stellar parameter."""
        
        star_masses = self.imf.sample(self.target_mass)
        self.total_mass_formed = star_masses.sum()
        
        self.stars.pars = self.isoc.get_stellar_pars_at(star_masses, self.logage, self.Z )
        self.nstars=self.stars.pars.shape[0]
        self.total_mass_current = (self.stars.pars['MASSACT'][np.isfinite(self.stars.pars['MASSACT'])]).sum()   

    def observe_stars(self,filterlist = None):
        """Obtain the spectra of each star using the stellar spectral library and convolve with the
        supplied list of filter transmission curves to obtain and populate the SED field of self."""

        if filterlist is not None:
            self.filterlist = filterlist
        self.nfilters = len(self.filterlist)
        self.band_names = observate.filter_dict(self.filterlist) 

        live = (np.where(np.isfinite(self.stars.pars['MASSACT'])))[0]
        self.ndead = self.nstars - live.shape[0]
        self.stars.seds = np.zeros([self.nstars,self.nfilters])
        self.stars.lbol = np.zeros(self.nstars)
        self.stars.seds[live,:], self.stars.lbol[live], self.integrated_spectrum = self.speclib.generateSEDs(self.stars.pars[live], filterlist, attenuator = None, intspec = True )

    def plot_CMD(self,iname,jname,kname, outfilename = None):
        """Plot the color-magnitude diagram of the stars, given the filter nicknames for
        the desired color and magnitude."""
        
        live = (np.where(np.isfinite(self.stars.pars['MASSACT'])))[0]
        pl.plot(self.stars.seds[live,self.band_names[iname]]-self.stars.seds[live,self.band_names[jname]],
                selfstars.seds[live,self.band_names[kname]],marker='o',linestyle = 'None',alpha = 0.5)
        pl.ylim(10,-5)
        pl.xlabel(r'%s - %s' %(iname,jname))
        pl.ylabel(r'%s' % kname)
