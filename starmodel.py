import os, glob
import numpy as np
#import astropy.constants as constants

import observate
from modelgrid import *


class BaSeL3(SpecLibrary):
    """Class to handle BaSeL 3.1 stellar spectral library"""
    specdir = os.getenv('hrdspy')+'/data/spectra/'
    flux_unit_mod = 'erg/s/AA'
    flux_unit_out = 'erg/s/AA/cm^2 at a distance of 10pc'
    output_to_model_units = 10**( np.log10(4.0*np.pi)+2*np.log10(pc*10) )
    
    def __init__(self):
        self.pars = None
        self.spectra = None

    def read_all_Z(self):
        """Read the spectral libraries (as fits binary tables) for all avaliable metallicities,
        store in a SpecLibrary object."""
        flist = glob.glob(self.specdir+'wlbc*.fits')
        if len(flist) == 0 : raise NameError('nothing returned for ls ',self.specdir,'wlbc*.fits')
            
        self.Z_list=[]
        self.Z_legend = []
        for i,f in enumerate(flist):
            Z=f.split('_')[1]
            self.Z_legend.append(Z)
            self.Z_list.append( (10** (float( Z.replace('m','-').replace('p','') )) ) * 0.0190) #convert to an absolute metallicity
            self.read_one_Z(self.Z_legend[i])
        self.Z_list=np.array(self.Z_list)

    def read_one_Z(self,inZ):
        """Read the spectral library from a binary fits file for only one metallicity, specified as
        either a float of absolute metallicity (0.019 = Z_sun) or as the BaSeL metallicity descriptor,
        e.g m10 for log(Z/Z_sun) = -1"""
        if type(inZ) is float :
            Zname = 'p%02.0f' % (np.log10(inZ/0.0190)*10)
            if inZ < 0:
                Zname=Zname.replace('p','m')
            
        elif type(inZ) is str :
            Zname = inZ
            inZ = (10** (float( inZ.replace('m','-').replace('p','') )) ) * 0.0190

        
        filename = glob.glob(self.specdir+'wlbc99_'+Zname+'_cor.fits')
        parnames=['LOGL','LOGG','LOGT','M_H','V_TURB','XH','ID']
                
        wave, spec, pars = self.read_model_from_fitsbinary(filename[0], parnames, wavename = 'WAVELENGTH')
        zz=np.zeros(pars.shape[0],dtype= np.dtype([('Z','<f8')]))
        zz['Z'] = inZ
        pars=self.join_struct_arrays([pars,zz])
        if self.pars is None:
            self.pars = pars
            self.wavelength = wave
            self.spectra = spec
        else:
            self.pars = np.hstack([self.pars, pars])
            self.spectra = np.vstack([self.spectra, spec])
                        
    def spectra_from_pars(self, parstruct):
        return self.generate_spectrum(parstruct['LOGL'],parstruct['LOGT'],parstruct['LOGG'],parstruct['Z'])

    def generate_spectrum(self, logl, logt, logg, Z):
        logz = np.log10(Z/0.0190)
        spec = self.interpolate_to_pars( np.array([logt,logg,logz]).T, parnames = ['LOGT','LOGG','M_H'])

        log_lbol_lsun = np.log10(observate.Lbol(self.wavelength,spec)) - np.log10(lsun)
        renorm = 10**(logl - log_lbol_lsun)
        return (renorm * spec.T).T / self.output_to_model_units #renormalize and divide by area (in cm^2) of 10pc sphere
        
