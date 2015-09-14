### Classes for fitting stellar models to obseved SEDs.
### Based on the pyxydust module

import os, time
import numpy as np
import astropy.io.fits as pyfits

from sedpy import observate
import starmodel, isochrone, catio
import hrdspyutils as utils

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
        self.model_filterlist = observate.load_filters(self.rp['model_fnamelist']) #filter objects
        self.fit_filterlist = observate.load_filters(self.rp['fit_fnamelist'])
        
    def load_data(self):
        """Read the catalogs, apply distance modulus,
        and determine 'good' pixels"""
        self.data_filterlist = observate.load_filters(self.rp['data_fnamelist'])
        self.data_mag, self.data_magerr, self.data_header = catio.load_image_cube(self.rp)
        self.distance_modulus = 5.0*np.log10(self.rp['dist'])+25
        self.nobj = self.data_mag.shape[0]
                            
    def setup_output(self):
        """Create arrays to store fit output for each pixel."""
        self.max_lnprob = np.zeros(self.nobj)+float('NaN')
        self.parval ={}
        for parn in self.rp['outparnames']:
            self.parval[parn] = np.zeros([self.nobj,
                                          len(self.rp['percentiles'])+1])+float('NaN')
        if self.rp['return_residuals'] is True:
            self.delta_best = {}
            for fname in self.rp['fit_fnamelist']:
                self.delta_best[fname] = np.zeros([self.nobj])+float('NaN')
        self.stargrid.parorder = utils.sortgrid(self.stargrid.pars,
                                            sortlist = self.rp['outparnames'])

    def fit_image(self):
        """Fit every 'pixel' in an image."""
        if hasattr(self,'max_lnprob') is False:
            self.setup_output()

        #build matching arrays of observed and and model SED
        self.stargrid.sed = np.array([ self.stargrid.pars[band] for band in
                                       self.rp['fit_fnamelist']]).T
        obs  = (np.array([self.data_mag[band] for band in self.rp['fit_fnamelist']]) -
                self.distance_modulus)
        err = np.array([self.data_magerr[band+'_unc'] for band in self.rp['fit_fnamelist']])
        
        start = time.time()
        for ipix in xrange(self.nobj):
            self.fit_pixel(ipix, obs[:,ipix], err[:,ipix])
        duration = time.time()-start
        print('Done all pixels in {0:.1f} seconds'.format(duration) )


class StarfitterGrid(Starfitter):

    def fit_pixel(self, ipix, obs, err, store = True, show_cdf = False):
        """Determine -\chi^2/2 of every model for a given pixel, and store moments
        of the CDF for each parameter as well as the bestfitting model parameters.
        Optionally store magnitude residuals from the best fit."""
        mask = (obs != 0) & np.isfinite(obs) & np.isfinite(err)
        lnprob , delta_mag = utils.lnprob_grid(self.stargrid, obs, err, mask)
        self.store_percentiles(ipix, lnprob, delta_mag)
        
    def store_percentiles(self, ipix, lnprob, delta_mag, tiny_lnprob = -1e30):
        """Store percentiles of the marginalized pdf.
        The sorting of each parameter in stargrid should be done
        prior to this function, since this is a time sink when the grid is large"""

        lnprob[~isfinite(lnprob)] = tiny_lnprob
        lmax = lnprob.max()
        if lmax <= tiny_lnprob:
            print(ix,iy)
            return
        ind_max = np.argmax(lnprob)
        self.max_lnprob[ipix] = lmax
        for i,fname in enumerate(self.rp['fit_fnamelist']):
            self.delta_best[fname][ipix] = delta_mag[ind_max,i]

        for i, parn in enumerate(self.rp['outparnames']):
            par = squeeze(self.stargrid.pars[parn])
            order = self.stargrid.parorder[parn]
            cdf = cumsum(exp(lnprob[order])) / np.sum(exp(lnprob))
            ind_ptiles= np.searchsorted(cdf,self.rp['percentiles'])
            # should linear interpolate instead of average.
            self.parval[parn][ipix,:-1] = (par[order[ind_ptiles-1]] +par[order[ind_ptiles]])/2.0 
            self.parval[parn][ipix,-1] = par[ind_max]        
    
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
        sed, self.stargrid.lbol, tmp = self.basel.generateSEDs(self.stargrid.pars,self.model_filterlist,
                                                               attenuator = attenuator,
                                                               wave_min=self.rp['wave_min'],
                                                               wave_max=self.rp['wave_max'])
        #add SED absolute magnitudes to stargrid parameters
        dt = zip(self.rp['model_fnamelist'], ['float64']*len(self.rp['model_fnamelist']))
        self.stargrid.pars = self.stargrid.join_struct_arrays([self.stargrid.pars, squeeze(sed.view(dtype = dt))])
                
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

    def fit_pixel(self, ipix, mag, err):
        obs = {}
        obs['maggies'] = 10**(0-mag/2.5)
        obs['ivar'] = (obs['maggies']*err/1.086)**(-2)
        obs['mask'] = ((mag < 0) & np.isfinite(mag))

        sampler = self.sample(obs, theta_names)

    def sample(self,obs, theta_names):
        initial = self.initial_proposal(theta_names, obs = obs)

        #get a sampler, burn it in, and reset
        sampler = emcee.EnsembleSampler(self.rp['nwalkers'], self.rp['ndim'], self.lnprob, threads=nthreads,
                                        args = [obs, theta_names] )
        pos,prob,state,blob = sampler.run_mcmc(initial, self.rp['nburn'])
        sampler.reset()

        #cry havoc
        sampler.run_mcmc(np.array(pos),self.rp['nsteps'], rstate0=state)

        return sampler

    def initial_proposal(self, theta_names, obs = None):
        tnames = ['LOGT', 'LOGL', 'Z', 'LOGG', 'A_V', 'R_V', 'F_BUMP', 'UV_SLOPE']
        logl = c1 * obs['maggies'].avg()
        logt = c2 * (obs['maggies'][0]/obs['maggies'][-1])
        
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
    
