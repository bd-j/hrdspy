#Example hrdspy usage.  determine dispersion in integrated spectrum
#as a function of wavelength for a given mass and age.

import matplotlib.pyplot as pl
import time, pickle
import numpy as np
import starmodel, isochrone, cluster, imf
from sedpy import observate, attenuation

# choose a few filters and load them
filterlist = observate.load_filters(['bessell_B','bessell_R'])

# instantiate and load the isochrones and spectra first so you
# don't have to do it for each cluster realization
isoc = isochrone.Padova2007()
isoc.load_all_isoc()
speclib = starmodel.BaSeL3()
speclib.read_all_Z()
IMF = imf.SalpeterIMF(mlo=[0.1], mhi=[100.], alpha=[2.35])

A_v = 0.0
wave = speclib.wavelength
nw = wave.shape[0]

hdrtxt = ("A list of [header, wave, spectrum, stellar_masses] for {0} realizations "
          "of a cluster with M_*={1} M_sun of stars drawn from a Salpeter IMF (0.1, 100) "
          "at logt={2} yrs and metallicity Z={3} with A_v={4}.  Spectrum is "
          "a {0} x {5} array while stellar masses is a {0} element list of lists, "
          "where each sublist gives the stellar masses above {6} M_sun for one realization.")
nametemplate = "stochastic_lib/salp_stoch{0}_logM{1:3.1f}_logt{2:4.2f}.p"


def make_realizations(mtot, logage, nreal=10, mlim=1.0, Z=0.0190):
    """Draw stochastic realizations of an SSP with a given mass and
    age.

    :param mtot:
        Total stellar mass of the cluster in M_sun.
        
    :param logage:
        Log_10 of the age of the cluster in years.
        
    :param nreal: (default:10)
        Number of stochastic realizations of the cluster to make.
    """
    header = (hdrtxt.format(nreal, mtot, logage, Z, A_v, nw, mlim))
    name = nametemplate.format(nreal, np.log10(mtot), logage)
    # set up output
    spectrum = np.zeros([nreal, nw])
    cluster_values = np.zeros([nreal, 3])
    masses = []

    for i in xrange(nreal):

        #use Padova2007, BaSeL3.1, and Salpeter IMF (default)
        cl = cluster.Cluster(mtot, logage, Z, isoc = isoc,
                             speclib = speclib, IMF = None)
        cl.generate_stars()
        cl.observe_stars( filterlist )
        spectrum[i,:] = cl.integrated_spectrum
        thismasses = cl.stars.pars['MASSIN']
        print('{0} stars with M_ini > {1}M_sun'.format((thismasses > mlim).sum(), mlim))
        masses += [thismasses[thismasses > mlim]]
        cluster_values[i,0] = cl.total_mass_formed
        cluster_values[i,1] = cl.nstars
        cluster_values[i,2] = cl.ndead
        #write output
    with open(name, "wb") as f:
        pickle.dump( [header, wave, spectrum, masses], f)


if __name__=="__main__":
    
    #masses = 10**(4.5 + np.arange(3) *0.5)
    masses = [10**5.0]
    nreal, Z, mlim = 100, 0.0190, 2.0
    ages = 7.5 + np.arange(10) * 0.25
    #ages = [9.5, 9.75]
    for mtot in masses:
        for logage in ages:
            make_realizations(mtot, logage, nreal=nreal, Z=Z, mlim=mlim)

  
    
