import numpy as np
import pyfits
from StringIO import StringIO
from astropy.io import ascii

import observate


#######
### function to obtain likelihood ####
#######

def lnprob_grid(grid, obs, err, mask):
    #linearize fluxes.  
    inds = np.where(mask > 0)
    inds=inds[0]
    mod_maggies = 10**(0-grid.sed[...,inds]/2.5)
    obs_maggies = 10**(0-obs[inds]/2.5)
    obs_ivar = (obs_maggies[inds]*err[inds]/1.086)**(-2)

    #best scale for these parameters.  sums are over the bands
    #lstar = np.squeeze(( (mod_maggies*obs_maggies*obs_ivar).sum(axis=-1) ) /
    #                   ( ( (mod_maggies**2.0)*obs_ivar ).sum(axis=-1) ))
    #lbol = grid.lbol*lstar
        
    #probability with dimensional juggling to get the broadcasting right
    #chi2 =  (( (lstar*mod_maggies.T).T - obs_maggies)**2)*obs_ivar
    #delta_mag = 0-2.5*np.log10((lstar*mod_maggies.T).T/obs_maggies)

    chi2 = (( mod_maggies - obs_maggies)**2)*obs_ivar
    lnprob = np.squeeze(-0.5*chi2.sum(axis=-1))
    delta_mag = 0-2.5*np.log10(mod_maggies/obs_maggies)
    #clean NaNs
    
    return lnprob, delta_mag

#not working yet....
def cdf_moment(par,lnprob,percentiles,save=False,plot=False):
    order = np.argsort(par)
    cdf = np.cumsum(np.exp(lnprob[order])) / np.sum(np.exp(lnprob))
    ind_ptiles= np.searchsorted(cdf,percentiles)
    ind_max=np.argmax(lnprob_isnum)

    return np.concatenate(par[order[ind_ptiles]],par[ind_max])

def prob_less_than(par, lnprob, values):
    pass

##########
#### Data Inputs ####
#########

def load_image_cube(rp):
    """Convert input catalogs into 1 x nstar x nfilter arrays for the fitter"""
    
    if rp['source'] is 'mcps':
        optical = read_some_mcps(**rp)
        header = optical[['RAh','Dec','flag']]
    elif rp['source'] is 'massey':
        optical = read_massey(**rp)
        header = optical[['RAh','Dec','spType']]

    nstar = len(optical)
    #    galex = (mcps['RAh'], mcps['Dec'])

    data_mag = np.zeros( [1, nstar, len(rp['fnamelist'])] )
    data_magerr = np.zeros( [1, nstar, len(rp['fnamelist'])] )
    print('ok')
    for i,fname in enumerate(rp['fnamelist']):
        filt = observate.Filter(fname)
        if fname in optical.dtype.names:
            data_mag[...,i] = optical[fname]-filt.ab_to_vega
            data_magerr[...,i] = optical[fname+'_unc']
        #elif fname in galex.dtype.names:
        #    data_mag[...,i] = galex[fname]
        #    data_magerr[...,i] = galex[fname+'_unc']
    return data_mag, data_magerr, header

def read_some_mcps(catalog_name = 'table1.dat', lines = None, nlines = 100, fsize = 24.1e6, **extras):
    """Read in just a selection of lines from the full MCPS catalog"""

    bytes_per_line = 74 #for the MCPS LMC table1.dat
    dt = {'names':('RAh','Dec',
                   'bessell_U','bessell_U_unc',
                   'bessell_B','bessell_B_unc',
                   'bessell_V','bessell_V_unc',
                   'bessell_I','bessell_I_unc',
                   'flag'),
          'formats':('<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<i4')}
    if lines is None:
        print('randomly choosing {0} lines'.format(nlines))
        lines = np.sort(np.random.uniform(0, fsize, nlines).astype(int))
    
    f = open(catalog_name, 'r')
    somelines = ''
    for lno in lines:
        f.seek(lno * bytes_per_line)
        somelines+=f.readline()
    f.close()
    somelines = StringIO(somelines)

    data = np.loadtxt(somelines, dtype = dt)
    return data

def read_massey(catalog_name = None, spec_types = True, spec_catalog = None, **extras):
    """Read in data from Massey 2002, optionally only include stars with spectral types."""

    table = ascii.read(catalog_name)
    if spec_types is True:
        sptable = ascii.read(spec_catalog)
        table = table[sptable['CNum']-1]  

    dt = {'names':('RAh','Dec',
                   'bessell_U','bessell_U_unc',
                   'bessell_B','bessell_B_unc',
                   'bessell_V','bessell_V_unc',
                   'bessell_R','bessell_R_unc',
                   'spType', 'r_spType'),
          'formats':('<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','a12','a8')}
    newt = np.zeros(len(table), dtype = dt)

    newt['RAh'] = (table['RAh'] + table['RAm']/60. + table['RAs']/3600.)
    newt['Dec'] = (-1)*(table['DEd'] + table['DEm']/60. + table['DEs']/3600.)
    newt['bessell_V'] = table['Vmag']
    newt['bessell_B'] = table['Vmag'] + table['B-V']
    newt['bessell_U'] = table['Vmag'] + table['B-V'] + table['U-B']
    newt['bessell_R'] = table['Vmag'] - table['V-R']
    newt['bessell_V_unc'] = table['e_Vmag']
    newt['bessell_B_unc'] = np.sqrt(table['e_Vmag']**2 + table['e_B-V']**2) #use np.hypot
    newt['bessell_U_unc'] = np.sqrt(table['e_Vmag']**2 + table['e_B-V']**2 + table['e_U-B']**2)
    newt['bessell_R_unc'] = np.sqrt(table['e_Vmag']**2 + table['e_V-R']**2)

    if spec_types is True:
        newt['spType'] = sptable['SpType']
        newt['r_spType'] = sptable['r_SpType']
    else:
        newt['spType'][sptable['CNum']-1] = sptable['SpType']
        newt['r_spType'][sptable['CNum']-1] = sptable['r_SpType']

    return newt


