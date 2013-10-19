#Example hrdspy usage.  determine dispersion in integrated spectrum
#as a function of wavelength for a given mass and age.

import numpy as np
import starmodel
import isochrone
import observate
import cluster
import matplotlib.pyplot as pl
import time

# choose a few filters and load them
filternamelist = ['ctio_mosaic_ii_Uj','ctio_mosaic_ii_B','ctio_mosaic_ii_V',
                  'ctio_mosaic_ii_Rc','ctio_mosaic_ii_Ic']
filterlist = observate.load_filters(filternamelist)
j, k = 1, 3 #color to plot


# instantiate and load the isochrones and spectra first so you
# don't have to do it for each cluster realization
isoc=isochrone.Padova2007()
isoc.load_all_isoc()
speclib = starmodel.BaSeL3()
speclib.read_all_Z()

# set cluster parameters
Z = 0.0076   #LMC metallicity
mtot = 1e3   #1000 solar masses
logage = np.arange(7.2,9.2,0.2) 
nreal = 10   #10 realizations of the cluster

# set up output
wave = speclib.wavelength
spectrum = np.zeros([nreal,wave.shape[0]])
cluster_values = np.zeros([nreal, 3])

pl.figure(1)

start =time.time()

for i in xrange(len(logage)):

    #use Padova2007, BaSeL3.1, and Salpeter IMF (default)
    cl = cluster.Cluster(mtot, logage[i], Z, isoc = isoc, speclib = speclib, IMF = None)
    cl.generate_stars()
    cl.observe_stars( filterlist )
    spectrum[i,:] = cl.integrated_spectrum
    cluster_values[i,0] = cl.total_mass_formed
    cluster_values[i,1] = cl.nstars
    cluster_values[i,2] = cl.ndead

    #cl.reset_stars()
    #cl.decompose()

    pl.plot(cl.stars.seds[:,j]-cl.stars.seds[:,k],cl.stars.seds[:,k],marker = 'o',
            linestyle = 'None',label = ('%4.2f' %logage[i]) )

s = time.time() -start
print("Done %i clusters in %f seconds" %(nreal,s))

pl.xlim(-1,2.5)
pl.xlabel('%s-%s' % (filterlist[j].nick,filterlist[k].nick) )
pl.ylim(4,-5)
pl.ylabel(filterlist[k].nick)
pl.legend()
pl.savefig('example_cmd.png')
    
    
