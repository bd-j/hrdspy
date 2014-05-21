import cmf
import numpy as np
import matplotlib.pyplot as pl

clusters = cmf.CMF()
bins = 10**np.linspace(6.7,8,14)
nsim = 100

bcent = np.log10(bins[0:-1]) + np.diff(np.log10(bins))/2.

pl.figure()
all_sfh = np.zeros([nsim, len(bins)-1])
sfr = 1e-2
for i in xrange(nsim):
    m, t, sfh = clusters.sfh(sfr = sfr, interval = np.max(bins), timebins = bins)
    all_sfh[i,:] = sfh
    #pl.plot(t,m,'.')
    pl.plot(bcent, sfh)

pl.xlabel(r'$\log t_{{lookback}}$')
pl.ylabel(r'SFR(t)')
pl.title(r'$\langle SFR \rangle = {0}$'.format(sfr))
pl.show()
