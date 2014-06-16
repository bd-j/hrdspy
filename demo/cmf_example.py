import imf
import numpy as np
import matplotlib.pyplot as pl

clusters = imf.CMF()
bins = 10**np.linspace(6.7,8,14)
nsim = 1000

bcent = np.log10(bins[0:-1]) + np.diff(np.log10(bins))/2.

pl.figure()
all_sfh = np.zeros([nsim, len(bins)-1])
sfr = 1e-2

for i in xrange(nsim):
    m, t, sfh = clusters.sfh(sfr = sfr, interval = np.max(bins), timebins = bins)
    all_sfh[i,:] = sfh
    #pl.plot(t,m,'.')
    pl.plot(bcent, sfh)

pl.figure()
pl.xlabel(r'$\log t_{{lookback}}$')
pl.ylabel(r'SFR(t)')
pl.title(r'$\langle SFR \rangle = {0}$'.format(sfr))

pl.figure()
b1 = 5
r = all_sfh[:,b1]/all_sfh[:,b1+1]
leg = r'$\langle SFR \rangle = {0}$'.format(sfr)
pl.hist(np.log10(r), bins = np.linspace(-2,2,40),
        alpha = 0.5, histtype = 'stepfilled', color = 'blue')#, legend = '')
pl.xlabel(r'$\log SFR_{{{0}}}/SFR_{{{1}}}$'.format(bcent[b1], bcent[b1+1]))
pl.show()
