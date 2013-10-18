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
        self.goodpix = np.where(gg.sum(axis = 2) == len(self.rp['fnamelist'])) #restrict to detections in all bands
                    
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
            self.delta_best = np.zeros([self.nx,self.ny,len(self.filterlist)])+float('NaN')

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
        
    def store_percentiles(self, iy, ix, lnprob, delta_mag):
        """Store percentiles of the marginalized pdf.
        The sorting of each parameter in stargrid should be done
        prior to this function = this is a time sink when the grid is large"""
        ind_isnum = np.where(np.isfinite(lnprob))[0]
        if ind_isnum.shape[0] == 0:
            print(ix,iy)
            return
        lnprob_isnum = lnprob[ind_isnum]
        ind_max = np.argmax(lnprob_isnum)
        
        self.max_lnprob[iy,ix] = np.max(lnprob_isnum)
        self.delta_best[iy,ix,:] = delta_mag[ind_isnum[ind_max],:]
        for i, parn in enumerate(self.outparnames):
            par = np.squeeze(self.stargrid.pars[parn])[ind_isnum]
            order = np.argsort(par)
            cdf = np.cumsum(np.exp(lnprob_isnum[order])) / np.sum(np.exp(lnprob_isnum))
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
        self.stargrid.wavelength = self.basel.wavelength
        duration=time.time()-start
        print('Model Grid built in {0:.1f} seconds'.format(duration))


    def write_catalog(self, outparlist = None):
        """Write fit results to a FITS binary table.  Too highly specialized"""
        if outparlist is None: outparlist = self.rp['outparnames']
        dm = 5.0*np.log10(self.rp['dist']) + 25

        #input magnitudes
        m = self.basel.structure_array(self.data_mag[0,:,:]+dm,
                                       self.rp['fnamelist'])
        #input magnitude errors
        me = self.basel.structure_array(self.data_magerr[0,:,:],
                                         ['{0}_unc'.format(f) for f in self.rp['fnamelist']])
        #best-fit chi^2
        cb = self.basel.structure_array(self.max_lnprob[0,:]*(-2),
                                         ['chibest'])
        #paramater percentiles 
        pst = []
        for i, par in enumerate(outparlist):
            print(par, self.parval[par][0,:,0:3].shape)
            print(['{0}_p{1:5.3f}'.format(par.replace('galex_',''), pt) for pt in self.rp['percentiles']])
            pst +=  [self.basel.structure_array(self.parval[par][0,:,0:3],
                                            ['{0}_p{1:03.0f}'.format(par.replace('galex_',''), pt*1000)
                                             for pt in self.rp['percentiles']])]
            pst += [self.basel.structure_array(self.parval[par][0,:,-1], ['{0}_best'.format(par.replace('galex_',''))])]
        #put everything together (including ra, dec, and other header info) and write it out
        cat = self.basel.join_struct_arrays( [self.rp['data_header'], m, me, cb] + pst )
        cols = pyfits.ColDefs(cat)
        tbhdu = pyfits.new_table(cols)
        tbhdu.writeto('{0}_starprops.fits'.format(self.rp['outname']), clobber = True)


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


    def set_params_from_isochrone(self, Z = None, logl_min = 1, logt_max = 4.59, logt_min =3.5):
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
        self.stargrid.pars['LOGL'] = isoc.pars['LOGL'][draw]+np.random.normal(0, 0.15, self.rp['ngrid'])
        self.stargrid.pars['LOGT'] = np.clip(isoc.pars['LOGT'][draw]+
                                             np.random.normal(0, 0.05, self.rp['ngrid']),
                                             logt_min, logt_max) #clip again for the stellar library
        self.stargrid.pars['LOGG'] = np.clip(isoc.pars['LOGG'][draw] + 1.05, -0.5, 4.8) #HACK. added one to better fit the basel library
        
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
        #should be list of dicts or dict of lists?  no, dict of dicts!
        self.params = {}

    def fit_pixel(self, ix, iy):
        obs, err  = self.data_mag[ix,iy,:], self.data_magerr[ix,iy,:]
        obs_maggies = 10**(0-obs/2.5)
        obs_ivar = (obs_maggies*err/1.086)**(-2)
        mask = np.where((obs < 0) & np.isfinite(obs) , 1, 0)

        sampler = self.sample(obs_maggies, obs_ivar, mask)

    def sample(self,obs_maggies, obs_ivar, mask):
        initial = self.initial_proposal()

        #get a sampler, burn it in, and reset
        sampler = emcee.EnsembleSampler(self.rp['nwalkers'], self.rp['ndim'], self.lnprob, threads=nthreads,
                                        args = [obs_maggies,obs_ivar,mask] )
        pos,prob,state,blob = sampler.run_mcmc(initial, self.rp['nburn'])
        sampler.reset()

        #cry havoc
        sampler.run_mcmc(np.array(pos),self.rp['nsteps'], rstate0=state)

        return sampler

    def initial_proposal(self):
        parnames = self.lnprob.lnprob_parnames
        theta = np.zeros(len(parnames))
        for j, parn in enumerate(parnames) :
            theta[:,j] = np.random.uniform(self.params[parn]['min'],self.params[parn]['max'])
        return theta


    def lnprob(self, theta, obs_maggies, obs_ivar, mask):
        lnprob_parnames = ['UMIN', 'UMAX', 'GAMMA', 'QPAH', 'MDUST']
        #pardict = {lnprob_parnames theta} #ugh.  need quick dict or struct_array from list/array

        #prior bounds check
        ptest=[]
        for i,par in enumerate(lnprob_parnames):
            ptest.append(pardict[par] >= self.params[par]['min'])
            ptest.append(pardict[par] <= self.params[par]['max'])
            if self.params[par]['type'] == 'log' : pardict[par] = 10**pardict[par]
                
        if False in ptest:
            #set lnp to -infty if parameters out of prior bounds
            lnprob = -np.infty
            lbol = -1

        else:
            #model sed (in AB absolute mag) for these parameters
            sed, lbol = self.model(**pardict)
            sed_maggies = 10**(0-sed/2.5)
        
            #probability
            chi2 = ( (sed_maggies - obs_maggies)**2 )*obs_ivar
            inds = np.where(mask > 0)
            lnprob = -0.5*chi2[inds].sum()
        
        return lnprob, [lbol]


    def model():
        pass

#####
#### Output methods
    
