import numpy as np
import imf
import starmodel
import isochrone
import matplotlib.pyplot as pl

class Cluster(object):
    def __init__(self,target_mass, logage, Z, isoc = None, IMF = None, speclib = None)
        self.target_mass = target_mass
        self.logage = logage
        self. Z = Z
        self.compose(isoc, IMF, speclib)
        
    def compose(self, IMF, isoc, speclib):
        #if IMF is None:
        self.imf  = imf.SalpeterIMF()
            #else:
            #self.imf = IMF
            #if isoc is None:
        self.isoc=isochrone.Padova2007()
        self.isoc.load_all_isoc()
            #else:
            #self.isoc = isoc
            #if speclib is None:
        self.speclib = starmodel.BaSeL3()
            # self.speclib.load_all_spectra()
            #else:
            #self.speclib = speclib

        self.stars = starmodel.SpecLibrary()
        #self.stars.wavelength = self.speclib.wavelength
        
    def generate_stars(self):
        star_masses = self.imf.sample(self.total_mass)
        self.total_mass_formed = star_masses.sum()
        
        self.stars.pars = self.isoc.get_stellar_pars_at(star_masses, self.logage, self.Z )
        self.nstars=self.stars.shape[0]
        self.total_mass_current = (self.stars.pars['MASSACT'][np.isfinite(self.stars.pars['MASSACT'])]).sum()   

    def observe_stars(self,filterlist = None):
        if filterlist is not None:
            self.filterlist = filterlist
        nf = len(self.filterlist)
        self.band_names = [f.nick for f in filterlist] #think about nmaking a dictionary with values = index of filter.  maybe add to observate

        live = (np.where(np.isfinite(self.stars.pars['MASSACT'])))[0]
        self.stars.seds = np.zeros(self.nstars,nf)
        self.lbol = np.zeros(self.nstars)
        self.stars.seds[live,:], self.stars.lbol[live], spjunk = self.speclib.generateSEDs(self.stars.pars[live], filterlist, attenuator = None )

    def plot_CMD(self,i,j,k, outfilename = None): #possibly allow input to be band_names
        live = (np.where(np.isfinite(self.stars.pars['MASSACT'])))[0]
        pl.plot(self.stars.seds[live,i]-selfstars.seds[live,j],selfstars.seds[live,k],marker='o',linestyle = 'None',alpha = 0.5)
        pl.ylim(10,-5)
        pl.xlabel(r'%s - %s' %(self.band_names[i],self.band_names[j]))
        pl.ylabel(r'%s' % self.band_names[k])
