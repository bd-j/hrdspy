import numpy as np
import os, time
import starfitter as sf
import attenuation, observate
import matplotlib.pyplot as pl
import plotter

#### SETTINGS
#root name for output files, distance in Mpc, number of SEDs in grid, min and max wavelngth for luminosity calculation
rp = {'outname': 'results/mcps_lmc_test', 'dist': 0.050, 'ngrid': 5e4, 'wave_min':92, 'wave_max':1e7}
#percentiles of the CDF to return
rp['percentiles'] = np.array([0.025, 0.5, 0.975])
#location of the MCPS catalog
rp['catalog_name'] = '/Users/bjohnson/SFR_FIELDS/Nearby/MCs/lmc/mcps/table1.dat'
#number of entries from MCPS catalog to fit
rp['nlines'] = 5000
#specific lines from the MCPS catlog to fit.  if None, randomly sample the catalog (Slow!!!)
rp['lines'] = None # np.arange(1000)+rp['nlines']

rp['dust_type'] = 'MilkyWay' #MilkyWay|LMC|SMC.  In the latter cases, R_V should be set to None
#MCPS catalog filter curves
rp['fnamelist'] = ['U','B','V','I']
rp['fnamelist'] = ['bessell_{0}'.format(f) for f in rp['fnamelist']] #this is where I miss IDL: 'bessell_'+x.  could do as numpy string array

#Parameters for which to return CDF moments
rp['outparnames'] = ['LOGL', 'LOGT','A_V', 'LOGM']#'R_V','F_BUMP', 'LOGM'] 
rp['return_residuals'] = True # False or True #return residuals from the best fit in each band


#### DO THE FITTING
# Initialize the dust attenuation curve and the stellar SED fitter
dust = attenuation.Attenuator(dust_type = rp['dust_type'])
fitter = sf.StarfitterGrid(rp)

#set up priors in the grid
fitter.initialize_grid(params = None)
fitter.set_params_from_isochrone(Z = [0.0077])
fitter.stargrid.pars['R_V'] = 3.0 #fix extra dust parameters (should marginalize over, but requires thought about prior)
fitter.stargrid.pars['F_BUMP'] = 0.5 #fix extra dust parameters (should marginalize over, but requires thought about prior)

fitter.build_grid(attenuator = dust)

## Set up for predicting the emission in any band
## Do this after running fitter.initialize_grid() but before
## running fitter.fit_image()
pred_fnamelist = ['galex_NUV', 'galex_FUV']
pred_filt = observate.load_filters(pred_fnamelist)
prediction_sed, tmp1, tmp2 = fitter.basel.generateSEDs(fitter.stargrid.pars,pred_filt,attenuator = dust, wave_min = 92, wave_max = 1e7)
#for i in xrange(len(pred_filt)-1) :
fitter.stargrid.add_par(prediction_sed[:,0] + 5.0*np.log10(rp['dist'])+25,pred_filt[0].name)
fitter.rp['outparnames']+= [pred_filt[0].name]

fitter.load_data()
#Chhange the number of lines to extract
#fitter.rp['nlines'] = 1e4
#fitter.rp['lines'] =None
#fitter.load_data()

fitter.fit_image()

fitter.write_catalog(outparlist = ['galex_NUV', 'LOGT','LOGL', 'A_V'])

gf = {'goodfit':(fitter.rp['data_header']['flag'] == 10) & (fitter.data_mag[0,:,2] +18.5 < 20.0),'glabel': 'V<20 & flag==10'}

plotter.plot_precision(fitter, PAR = 'LOGT', **gf)
plotter.plot_precision(fitter, PAR = 'A_V', **gf)
plotter.plot_precision(fitter, PAR = 'A_V', **gf,
                                versus = fitter.parval['LOGT'][0,:,1])
plotter.plot_precision(fitter, PAR = 'galex_NUV', **gf)

plotter.plot_pars(fitter, PAR1 = 'LOGT', PAR2 = 'LOGL', **gf)
plotter.plot_pars(fitter, PAR1 = 'LOGT', PAR2 = 'A_V', **gf)

plotter.residuals(fitter, bands = [0, 1, 2, 3], colors  = ['m','b','g','r'], **gf)
