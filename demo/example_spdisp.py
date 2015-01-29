#Example hrdspy usage.  determine dispersion in integrated spectrum
#as a function of wavelength for a given mass and age.

import numpy as np
import starmodel
import isochrone
from sedpy import observate
import cluster
import matplotlib.pyplot as pl
import time

# choose a few filters and load them
filternamelist = ['galex_NUV','sdss_u0','sdss_r0']
filterlist = observate.load_filters(filternamelist)

# instantiate and load the isochrones and spectra first so you
# don't have to do it for each cluster realization
isoc=isochrone.Padova2007()
isoc.load_all_isoc()
speclib = starmodel.BaSeL3()
speclib.read_all_Z()

# set cluster parameters
Z = 0.0190   #solar metallicity
mtot = 1e4   #10000 solar masses
logage = 6.6   #10 Myr
nreal = 10   #10 realizations of the cluster


regenerate=True

start =time.time()
if regenerate:
    # set up output
    wave = speclib.wavelength
    spectrum = np.zeros([nreal,wave.shape[0]])
    cluster_values = np.zeros([nreal, 3])

    for i in xrange(nreal):

        #use Padova2007, BaSeL3.1, and Salpeter IMF (default)
        cl = cluster.Cluster(mtot, logage, Z, isoc = isoc, speclib = speclib, IMF = None)
        cl.generate_stars()
        cl.observe_stars( filterlist )
        spectrum[i,:] = cl.integrated_spectrum
        cluster_values[i,0] = cl.total_mass_formed
        cluster_values[i,1] = cl.nstars
        cluster_values[i,2] = cl.ndead

        #cl.reset_stars()
        #cl.decompose()

        

    s = time.time() -start
    print("Done %i clusters in %f seconds" %(nreal,s))
    
    start = time.time()
    bigcl = cluster.Cluster(mtot*nreal, logage, Z, isoc = isoc, speclib = speclib, IMF = None)
    bigcl.generate_stars()
    bigcl.observe_stars( filterlist )
    s = time.time() -start
    print("Done big clusters in %f seconds" %(s))

pl.figure(1)
for i in range(nreal):
    pl.plot(wave,spectrum[i,:],alpha = 0.3, color='grey')
pl.xlim(2e3,1e4)
pl.xscale('log')
pl.xlabel(r'$\lambda(\AA)$')
pl.ylim(1e-8,1e-4)
pl.yscale('log')
pl.ylabel(r'$erg/s/cm^2/\AA$ at $10pc$')
pl.plot(wave, spectrum.mean(axis = 0), color='black',label =r'$\langle f_\lambda\rangle$, M$_*=$%3.0e' % (mtot), linewidth=2.0  )
    

pl.plot(wave,bigcl.integrated_spectrum/nreal,color='red',label = r'$f_\lambda/%s$, M$_*=$%3.0e' % (nreal, (mtot*nreal)) )

pl.legend(loc=0)
pl.title(r'$\log Age = %4.2f$' % logage)
pl.savefig('example4.png')
pl.close()
    
    
