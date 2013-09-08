import numpy as np
import os, time
import starfitter as sf
import attenuation, observate
import matplotlib.pyplot as pl
import plotter

#### SETTINGS
#  Distance in Mpc, number of SEDs in grid, min and max wavelength for logL calculation
rp = {'dist': 0.050, 'ngrid': 5e4, 'wave_min':92, 'wave_max':1e7}
# Percentiles of the CDF to return
rp['percentiles'] = np.array([0.025, 0.5, 0.975])

# Input optical catalog(s)
rp['source'] = 'massey'  #massey|mcps

# Optical catalog parameters
if rp['source'] is 'mcps':
    rp['catalog_name'] = '/Users/bjohnson/SFR_FIELDS/Nearby/MCs/lmc/mcps/table1.dat'
    # Number of entries from MCPS catalog to fit
    rp['nlines'] = 5000
    # Specific lines from the MCPS catalog to fit (it's big!).
    # if None, randomly sample the catalog (Slow!!!)
    rp['lines'] = None # np.arange(1000)+rp['nlines']
    rp['fnamelist'] = ['U','B','V','I']
    # Root name for output files
    rp['outname'] = 'results/mcps_lmc_test'
elif rp['source'] is 'massey':
    rp['catalog_name'] = '/Users/bjohnson/SFR_FIELDS/Nearby/MCs/lmc/massey02_lmc_table3a.dat'
    rp['spec_types'] = True
    rp['spec_catalog'] = '/Users/bjohnson/SFR_FIELDS/Nearby/MCs/lmc/massey02_lmc_table4.dat'
    rp['fnamelist'] = ['U','B','V','R']
    rp['outname'] = 'results/massey_lmc_test'

rp['galex_csv_filename'] = '/Users/bjohnson/SFR_FIELDS/Nearby/MCs/lmc/massey_galex_lmc_sptyped.csv'

rp['dust_type'] = 'MilkyWay' #MilkyWay|LMC|SMC.  In the latter cases, R_V should be set to None

#filter curves
rp['fnamelist'] = ['bessell_{0}'.format(f) for f in rp['fnamelist']] #this is where I miss IDL
rp['fnamelist'] = ['galex_NUV'] + rp['fnamelist']

#Parameters for which to return CDF moments
rp['outparnames'] = ['LOGL', 'LOGT', 'A_V', 'LOGM']#'R_V','F_BUMP', 'LOGM'] 
rp['return_residuals'] = True # False or True #return residuals from the best fit in each band


#### DO THE FITTING
# Initialize the dust attenuation curve and the stellar SED fitter
dust = attenuation.Attenuator(dust_type = rp['dust_type'])
fitter = sf.StarfitterGrid(rp)

#Set up priors in the grid
fitter.initialize_grid(params = None)
fitter.set_params_from_isochrone(Z = [0.0077], logt_min = 3.7, logl_min = 2, logt_max = 4.65) #for only bright hotter stas in the massey catalog
#fix extra dust parameters (should marginalize over, but requires thought about prior)
fitter.stargrid.pars['R_V'] = 3.0 
fitter.stargrid.pars['F_BUMP'] = 1.0

#Build the SEDs of the grid
fitter.build_grid(attenuator = dust)

#pl.scatter(fitter.stargrid.pars['LOGT'],fitter.stargrid.sed[:,0]-fitter.stargrid.sed[:,4], c = fitter.stargrid.pars['A_V'], alpha = 0.5)

#raise ValueError('test grid now')

## Set up for predicting the emission in any band
## Do this after running fitter.initialize_grid() but before
## running fitter.fit_image()
if ('galex_NUV' in rp['fnamelist']) is False:
    pred_fnamelist = ['galex_NUV', 'galex_FUV']
    pred_filt = observate.load_filters(pred_fnamelist)
    prediction_sed, tmp1, tmp2 = fitter.basel.generateSEDs(fitter.stargrid.pars,
                                                           pred_filt,attenuator = dust,
                                                           wave_min = 92, wave_max = 1e7)
    #for i in xrange(len(pred_filt)-1) :
    fitter.stargrid.add_par(prediction_sed[:,0] + 5.0*np.log10(rp['dist'])+25,pred_filt[0].name)
    fitter.rp['outparnames']+= [pred_filt[0].name]
else:
    fitter.rp['outname'] = fitter.rp['outname']+'withNUV'
    ind = rp['fnamelist'].index('galex_NUV')
    fitter.stargrid.add_par(fitter.stargrid.sed[:,ind] + 5.0*np.log10(rp['dist'])+25,rp['fnamelist'][ind])
    fitter.rp['outparnames']+= ['galex_NUV']
    

fitter.load_data()
#Chhange the number of lines to extract
#fitter.rp['nlines'] = 1e4
#fitter.rp['lines'] =None
#fitter.load_data()

fitter.fit_image()

fitter.write_catalog(outparlist = [ 'galex_NUV','LOGT','LOGL', 'A_V'])

####### PLOTTING #########

plotter.plot_sptypes(fitter)
pl.xlim(9,42)
pl.savefig(rp['outname']+'_sptype_byclass.png')
plotter.plot_sptypes(fitter, cpar = 'A_V')
pl.xlim(9,42)
pl.savefig(rp['outname']+'_sptype_byAV.png')

if 'galex_NUV' in rp['fnamelist']:
    plotter.plot_sptypes(fitter, cpar = fitter.rp['data_header']['fov_radius'])
    pl.xlim(9,42)
    pl.savefig(rp['outname']+'_sptype_byFOVradius.png')
    pl.xlim(9,42)
    plotter.plot_sptypes(fitter, cpar = fitter.rp['data_header']['NUV_exptime'])
    pl.xlim(9,42)
    pl.savefig(rp['outname']+'_sptype_byNUVexptime.png')
    


raise ValueError('time to plot')

gf = {'goodfit':(fitter.rp['data_header']['spType'] != '') & (fitter.max_lnprob[0,:]*(-2) < 100),
      'glabel': r'spTyped $\chi^2 < 100$'}
      #gf = {'goodfit':(np.char.find(fitter.rp['data_header']['spType'],'WN') >=0) & (fitter.max_lnprob[0,:]*(-2) < 100),
      #'glabel': r'WN $\chi^2 < 100$'}



plotter.plot_precision(fitter, PAR = 'LOGT', **gf)
#pl.savefig('logt_unc.png')

#plotter.plot_precision(fitter, PAR = 'A_V', **gf)
#plotter.plot_precision(fitter, PAR = 'A_V',versus = fitter.parval['LOGT'][0,:,1], **gf)

plotter.plot_precision(fitter, PAR = 'galex_NUV', **gf)

plotter.plot_pars(fitter, PAR1 = 'LOGT', PAR2 = 'LOGL', loc = 4, **gf)
pl.savefig(rp['outname']+'_logl_logt.png')
plotter.plot_pars(fitter, PAR1 = 'LOGT', PAR2 = 'A_V', **gf)

plotter.residuals(fitter, bands = [0, 1, 2, 3], colors  = ['m','b','g','r'], **gf)

