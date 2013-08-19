import numpy as np
import os, time
import starfitter as sf
import attenuation

rp = {'outname': 'results/mcps_lmc', 'dist': 0.0490, 'ngrid': 5e4,
      'wave_min':92, 'wave_max': 1e7, #AA, range for determination of LDUST
      'percentiles' : np.array([0.025,0.5,0.975]) #output percentiles of the cdf for each parameter
      }

rp['outparnames'] = ['LOGL', 'LOGT', 'LOGM', 'A_V', 'R_V','F_BUMP'] 
rp['return_residuals'] = True # False or True #return best fit residuals in each band
rp['fnamelist'] = ['U','B','V','I']
rp['fnamelist'] = ['bessell_{0}'.format(f) for f in rp['fnamelist']]
#Data Input parameters here


###RUN THE FITTING PROCESS

fitter = sf.StarfitterGrid(rp)
## read in the image cube given filenames in rp
fitter.load_data() 

fitter.initialize_grid(params = None)
fitter.set_params_from_isochrone(Z = 0.0077)
fitter.build_grid()

## Set up for predicting the emission in any band
## Do this after running fitter.initialize_grid() but before
## running fitter.fit_image()
#import observate
#predfilt = observate.load_filters(your_prediction_filternamelist)
#prediction_sed, tmp1, tmp2 = fitter.stargrid.generateSEDs(fitter.dustgrid.pars,pred_filt,wave_min = 92, wave_max = 1e7)
#for i in len(pred_filt) : fitter.dustgrid.add_par(prediction_sed[:,i],pred_filt[i].name())
#fitter.rp['outparnames']+= [pred_filt[i].name() for i in len(pred_filt)]


fitter.fit_image()

fitter.write_output()
