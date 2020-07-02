import glob, pickle, os
import numpy as np
import matplotlib.pyplot as pl
import fsps
import astropy.constants as const

sps = fsps.StellarPopulation(zcontinuous=1)
sps.params['sfh'] = 0
sps.params['imf_type'] = 0 # salpeter
sps.params['logzsol'] = 0

lsun = const.L_sun.cgs.value
pc = const.pc.cgs.value
to_cgs = lsun/(4*np.pi*(10*pc)**2)


nametemplate = "stochastic_lib/salp_stoch{0}_logM{1:3.1f}_logt{2:4.2f}.p"


ns = 5
#masses = [4.0, 4.5, 5.0, 5.5]
#ages = np.arange(3)*0.25 + 7.5
masses = [5.0]
ages = [10.0]
wmin, wmax = 1.5e3, 2e4
nsample = 10
sample = np.random.uniform(0,nsample, ns).astype(int)
for age in ages:
    w, fspec = sps.get_spectrum(tage=10**age/1e9, peraa=True)
    fspec *= w
    for mass in masses:
        filename = nametemplate.format(nsample, mass, age)
        if not os.path.exists(filename):
            continue
        with open(filename, 'rb') as f:
            header, wave, spec, starmass = pickle.load(f)

        spec *= wave[None,:]/10**mass
        mean_spec = spec.mean(axis=0)
        pspec = np.percentile(spec, [16, 50, 84], axis=0)
        fig, ax = pl.subplots()
        ax.plot(wave, mean_spec, label='mean',
                color='k', linewidth=1.0)
        ax.fill_between(wave, pspec[0,:], pspec[2,:], label='16th-84th',
                         facecolor='grey', alpha=0.3, )
        for i in sample:
            ax.plot(wave, spec[i,:], color='blue', linewidth=0.5, alpha = 0.3)
            
        ax.plot(w, fspec*to_cgs, label='FSPS',
                color='red', alpha = 0.3, linewidth=1.0)

        wrange = (wave > wmin) & (wave < wmax)
        fmin, fmax = pspec[:, wrange].min(), pspec[:, wrange].max()
        #ax.set_ylim(fmin*0.95, fmax*1.05)
        ax.set_ylim(1e-7, 1e-5)
        ax.set_ylabel(r'$\lambda f_\lambda /M_*$')
        ax.set_xlim(wmin, wmax)
        ax.set_title(r'$\log M={0}$, $\log t={1}$'.format(mass, age))
        ax.legend(loc=0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        fig.savefig(filename.replace('.p','.png'))
        #pl.close(fig)
