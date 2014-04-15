# Determine main sequence luminosity of a given stellar mass
# in a set of filters.  Also, determine main sequence lifetime
import numpy as np
import matplotlib.pyplot as pl
import isochrone
import starmodel
import observate

mlist = [3,4,6,10,25]

Z = [0.0077]
fnamelist = ['galex_FUV','galex_NUV']
filterlist = observate.load_filters(fnamelist)

basel = starmodel.BaSeL3()
basel.read_all_Z()
isoc = isochrone.Padova2007()
isoc.load_isoc(Z)
ms = (isoc.pars['LOGG'] < 7) & (isoc.pars['LOGG'] >= 3)

sed, lbol, tmp = basel.generateSEDs(isoc.pars[ms],filterlist,
                                    attenuator = None,
                                    wave_min = 91,
                                    wave_max = 1e7)

for j,f in enumerate(fnamelist):
    pl.figure()
    
    pl.scatter(isoc.pars['MASSIN'][ms], sed[:,j], c = isoc.pars['LOGG'][ms])
    pl.ylabel(f)
    pl.xlabel('Mass (initial)')
    pl.xlim(1,27)
    pl.xscale('log')
    pl.ylim(9,-9)
    pl.colorbar()
    for m in mlist:
        pl.axvline(m)
    for i in np.arange(-5,5):
        pl.axhline(i,linestyle = ':', color = 'k')
    
pl.figure()
pl.scatter(isoc.pars['MASSIN'][ms], isoc.pars['LOGAGE'][ms], c = isoc.pars['LOGG'][ms])
pl.xlim(0,25)
pl.ylabel('log Age')
pl.xlabel('Mass (initial)')
pl.colorbar()
pl.xlim(1,27)
pl.xscale('log')
for m in mlist:
    pl.axvline(m)
for i in np.arange(6,9,0.2):
    pl.axhline(i,linestyle = ':', color = 'k')


pl.figure()
pl.scatter(isoc.pars['MASSIN'][ms], isoc.pars['LOGT'][ms], c = isoc.pars['LOGG'][ms])
pl.xlim(0,25)

pl.show()
