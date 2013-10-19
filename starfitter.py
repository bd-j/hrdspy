### Classes for fitting stellar models to obseved SEDs.
### Based on the pyxydust module

import os, time
import numpy as np
import astropy.io.fits as pyfits

import observate
import starmodel
import isochrone
import hrdspyutils as utils
import catio

from numpy import isfinite, exp, squeeze, cumsum
from hrdspyutils import lnprob_grid

class Starfitter(object):

    doresid = False

    def __init__(self, rp):
        self.rp = rp
        self.load_models()
        self.set_default_params()

    def load_models(self):
        """Load the BaSeL basis models, initialize the grid to hold resampled models,
        and load the filters"""
        self.basel = starmodel.BaSeL3()  # BaSeL 3.1 Basis
        self.basel.read_all_Z()
        self.stargrid = starmodel.SpecLibrary() #object to hold the model grid
        self.filterlist = observate.load_filters(self.rp['fnamelist']) #filter objects
        
    def load_data(self):
        """Read the catalogs, apply distance modulus,
        and determine 'good' pixels"""
        self.data_mag, self.data_magerr, self.rp['data_header'] = catio.load_image_cube(self.rp)
        dm = 5.0*np.log10(self.rp['dist'])+25
        self.data_mag = np.where(self.data_mag > 10.0, self.data_mag-dm, float('NaN'))
        self.nx, self.ny = self.data_mag.shape[0], self.data_mag.shape[1]
        
        gg = np.where((self.data_mag != 0) & np.isfinite(self.data_mag),1,0)
        #restrict to detections in all bands
        self.goodpix = np.where(gg.sum(axis = 2) == np.array(self.rp['use_filter_in_fit']).sum()) 
                    
    def setup_output(self):
        """Create arrays to store fit output for each pixel."""
        self.max_lnprob = np.zeros([self.nx,self.ny])+float('NaN')
        self.outparnames = self.rp['outparnames']
        self.parval ={}
        for parn in self.outparnames:
            self.parval[parn] = np.zeros([self.nx,self.ny,len(self.rp['percentiles'])+1])+float('NaN')

        try:
            self.doresid = self.rp['return_residuals']
        except (KeyError):
            self.doresid = False
        if self.doresid is True:
            self.delta_best = np.zeros([self.nx,self.ny,np.array(self.rp['use_filter_in_fit']).sum()])+float('NaN')
        self.stargrid.parorder = utils.sortgrid(self.stargrid.pars,
                                            sortlist = self.outparnames)

    def fit_image(self):
        """Fit every 'pixel' in an image."""
        if hasattr(self,'max_lnprob') is False:
            self.setup_output()
        start = time.time()
        for ipix in xrange(self.goodpix[0].shape[0]):
            iy, ix  = self.goodpix[0][ipix], self.goodpix[1][ipix]
            self.fit_pixel(ix,iy)
        duration =  time.time()-start
        print('Done all pixels in {0:.1f} seconds'.format(duration) )


class StarfitterGrid(Starfitter):

    def fit_pixel(self, ix, iy, store = True, show_cdf = False):
        """Determine \chi^2 of every model for a given pixel, and store moments
        of the CDF for each parameter as well as the bestfitting model parameters.
        Optionally store magnitude residuals from the best fit."""
        
        obs, err = self.data_mag[iy,ix,:], self.data_magerr[iy,ix,:]
        mask = np.where((obs != 0) & np.isfinite(obs), 1, 0)
    
        lnprob , delta_mag = utils.lnprob_grid(self.stargrid, obs, err, mask)

        self.store_percentiles(iy, ix, lnprob, delta_mag)
        
    def store_percentiles(self, iy, ix, lnprob, delta_mag, tiny_lnprob = -1e30):
        """Store percentiles of the marginalized pdf.
        The sorting of each parameter in stargrid should be done
        prior to this function, since this is a time sink when the grid is large"""

        ind_isnum = np.where(isfinite(lnprob))[0]
        if ind_isnum.shape[0] == 0:
            print(ix,iy)
            return
        lnprob[~isfinite(lnprob)] = tiny_lnprob
        ind_max = np.argmax(lnprob)
        self.max_lnprob[iy,ix] = lnprob.max()
        self.delta_best[iy,ix,:] = delta_mag[ind_max,:]

        for i, parn in enumerate(self.outparnames):
            par = squeeze(self.stargrid.pars[parn])
            order = self.stargrid.parorder[parn]
            cdf = cumsum(exp(lnprob[order])) / np.sum(exp(lnprob))
            ind_ptiles= np.searchsorted(cdf,self.rp['percentiles'])
            # should linear interpolate instead of average.
            self.parval[parn][iy,ix,:-1] = (par[order[ind_ptiles-1]] +par[order[ind_ptiles]])/2.0 
            self.parval[parn][iy,ix,-1] = par[ind_max]        
    
    def store_samples(self, lnprob, delta_mag, nsample):
        """resample the prior grid according to the likelihood so as to generate a sampling
        of the posterior-pdf"""
        pass

    def store_pdf(self, lnprob, delta_mag, at_parvals):
        """Calculate the CDF p(<p1 | p2, p3, p4) for given values of p1, p2,..."""
        pass
    
    def build_grid(self, attenuator = None):
        """Build the SED fitting grid using the intialized parameters."""
        start = time.time()
        self.stargrid.sed, self.stargrid.lbol, tmp = self.basel.generateSEDs(self.stargrid.pars,self.filterlist,
                                                                             attenuator = attenuator,
                                                                             wave_min=self.rp['wave_min'],
                                                                             wave_max=self.rp['wave_max'])
        #add SED absolute magnitudes to stargrid parameters
        sed = self.stargrid.sed.view(dtype = zip(self.rp['fnamelist'], ['float64']*len(self.rp['fnamelist'])))
        self.stargrid.pars = self.stargrid.join_struct_arrays([self.stargrid.pars, squeeze(sed.copy())])
        #now keep just those filters to be used in the fitting
        self.stargrid.sed = self.stargrid.sed[:,np.array(self.rp['use_filter_in_fit'])]
        
        self.stargrid.wavelength = self.basel.wavelength
        duration=time.time()-start
        print('Model Grid built in {0:.1f} seconds'.format(duration))


    def initialize_grid(self, params = None):
        """Draw grid parameters from prior distributions and build the grid."""
        if params is not None:
            self.params = params
        parnames = self.params.keys()
        theta = np.zeros([self.rp['ngrid'],len(parnames)])
        for j, parn in enumerate(parnames) :
            theta[:,j] = np.random.uniform(self.params[parn]['min'],self.params[parn]['max'],self.rp['ngrid'])
            if self.params[parn]['type'] == 'log':
                theta[:,j]=10**theta[:,j] #deal with uniform log priors   
        self.stargrid.set_pars(theta, parnames)


    def set_params_from_isochrone(self, Z = None, logl_min = 1, logt_max = 4.69, logt_min =3.5):
        """Draw model grid parameters from isochrones. This effectively uses the
        isochrone grid points as priors on the stellar properties."""
        isoc = isochrone.Padova2007()
        if Z is None:
            isoc.load_all_isoc()
        else:
            isoc.load_isoc(Z)
        good = ((isoc.pars['PHASE'] != 6) & #remove post-AGB
                (isoc.pars['LOGT'] < logt_max) & #remove stars hotter than exist in library
                (isoc.pars['LOGL'] > logl_min) &   #remove faint undetectable things (log solar luminosities)
                (isoc.pars['LOGT'] > logt_min) ) #remove very cool things
        isoc.pars = isoc.pars[good]

        #The draw should actually include mass weighting and completeness, but for now...
        #indeed, there should be a way to attach a prior probablity based on mass and completeness
        #or selection within a color/magnitude bin
        draw = (np.random.uniform(0, isoc.pars.shape[0]-1, self.rp['ngrid'])).astype(int)
        #add a little uncertainty to the isochrones
        self.stargrid.pars['LOGL'] = isoc.pars['LOGL'][draw]+np.random.normal(0, 0.15, self.rp['ngrid'])
        #clip again for the stellar library
        self.stargrid.pars['LOGT'] = np.clip(isoc.pars['LOGT'][draw]+
                                             np.random.normal(0, 0.05, self.rp['ngrid']),
                                             logt_min, logt_max)
        #HACK. added to better fit the basel library
        self.stargrid.pars['LOGG'] = np.clip(isoc.pars['LOGG'][draw]*1.18 + 0.2, -0.5, 4.9)
        # make Hot things have available 
        self.stargrid.pars['LOGG'][self.stargrid.pars['LOGT'] > 4.59] = 5.0
        #mass goes along for the ride.  Should actually come from interpolation of L and T
        #but this effectively includes some uncertainty in the isochrones
        self.stargrid.add_par(np.log10(isoc.pars['MASSIN'][draw]), 'LOGM') 

    def set_default_params(self):
        """Set the default model parameter properties."""
        self.params = {}
        self.params['LOGT'] = {'min': 3.5, 'max':4.8, 'type':'linear'}
        self.params['LOGG'] = {'min': -1, 'max':5.6, 'type':'linear'}
        self.params['LOGL'] = {'min': 3.3, 'max':4.8, 'type':'linear'}
        self.params['Z'] = {'min':0.0077, 'max':0.0077, 'type':'linear'}
        self.params['A_V'] = {'min':-1.5, 'max':0.5, 'type':'log'}
        self.params['R_V'] = {'min':2., 'max':4., 'type':'linear'}
        self.params['F_BUMP'] = {'min':0.1, 'max':1.1, 'type':'linear'}
        

class StarfitterMCMC(Starfitter):
    """Use emcee to do MCMC sampling of the parameter space for a given pixel.  Wildly unfinished/untested"""

    def set_default_params(self, large_number = 1e15):
        """Set the default model parameter ranges."""
        pass

    def fit_pixel(self, ix, iy):
        obs = {}
        mag, err  = self.data_mag[ix,iy,:], self.data_magerr[ix,iy,:]
        obs['maggies'] = 10**(0-mag/2.5)
        obs['ivar'] = (obs['maggies']*err/1.086)**(-2)
        obs['mask'] = ((mag < 0) & np.isfinite(mag))

        sampler = self.sample(obs)

    def sample(self,obs):
        initial = self.initial_proposal(obs = obs)

        #get a sampler, burn it in, and reset
        sampler = emcee.EnsembleSampler(self.rp['nwalkers'], self.rp['ndim'], self.lnprob, threads=nthreads,
                                        args = [obs, theta_names] )
        pos,prob,state,blob = sampler.run_mcmc(initial, self.rp['nburn'])
        sampler.reset()

        #cry havoc
        sampler.run_mcmc(np.array(pos),self.rp['nsteps'], rstate0=state)

        return sampler

    def initial_proposal(self, obs = None, theta_names = None):
        
        theta = np.zeros(len(parnames))
        for j, parn in enumerate(parnames) :
            theta[:,j] = np.random.uniform(self.params[parn]['min'],self.params[parn]['max'])
        return theta


    def lnprob(self, theta, obs, theta_names):
        pars = theta.view(dtype = zip(parnames,['float64']*len(parnames)))
        lnp_prior = prior_lnprob(pars)

        if ~np.isfinite(lnp_prior):
            return -np.infty
        else:
            #model sed (in AB absolute mag) for these parameters
            sed, lbol = self.model(pars)
            if lbol == 0.:
                return -np.infty #model parameters outside available grid
            sed_maggies = 10**(0-sed/2.5)
        
            #probability
            d = (sed_maggies - obs['maggies'])
            chi2 = ( d*d )*obs['ivar']
            inds = (obs['mask'] > 0)
            lnprob = -0.5*chi2[inds].sum() + lnp_prior
        
        return lnprob


    def model(pars):
        sed, lbol, tmp = self.basel.generateSEDs(pars,self.filterlist,
                                                 attenuator = self.attenuator,
                                                 wave_min=self.rp['wave_min'],
                                                 wave_max=self.rp['wave_max'])
        return sed, lbol


    def prior_lnprob(pars):
        
        #prior bounds check
        ptest=[]
        for i,par in enumerate(pars.dtype.names):
            ptest.append(pardict[par] >= self.params[par]['min'])
            ptest.append(pardict[par] <= self.params[par]['max'])
            if self.params[par]['type'] == 'log' : pardict[par] = 10**pardict[par]
        if False in ptest:
            #set lnp to -infty if parameters out of prior bounds
            lnprob = -np.infty
            lbol = -1


#####
#### Output methods
    
