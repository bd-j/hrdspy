import os, glob
import numpy as np
from sedpy.modelgrid import *

#import utils
#import sedmodels

class Isochrone(ModelLibrary):
    
    def __init__(self):
        pass
    
    def get_stellar_pars_at(self, initial_masses, logage, Z=0.019,
                            silent = False):
        """Given an array of initial masses, an age (in log yrs) and a
        metallicity, interpolate the isochrone stellar parameter
        values to these masses, age, and metallicity.  Interpolation
        in age and metallicity is nearest neighbor. Interpolation in
        mass uses piecewise linear weights in log(mass).
        """
        # Could interpolate with Delaunay Triangulation, but using
        # faster and safer nearest neighbor for now
        this_age = self.logage_list[self.nearest_index(self.logage_list, logage)]
        this_Z = self.Z_list[self.nearest_index(self.Z_list, Z)]
        if silent is False:
            print("Padova2007: looking for stars with "
                  "log(age) = {0} and Z = {1}".format(this_age, this_Z))
        #print(this_age, this_age.shape, this_Z, this_Z.shape)
        inds = np.where(np.logical_and(self.pars['LOGAGE'] == this_age,
                                       self.pars['Z'] == this_Z))
        inds = inds[0]
        
        # Now interpolate in (log) mass, keeping weights
        outinds, weights = self.weights_1DLinear(np.log(self.pars['MASSIN'][inds]),
                                                 np.log(initial_masses))
        i1 = inds[outinds]
        dead = (np.where(outinds[:,0] == outinds[:,1]))[0]

        #should loop this over a parameter name list with
        #structure_array to assemble output
        parnames = self.pars.dtype.names
        pars = np.zeros([weights.shape[0],len(parnames)])
        for i,p in enumerate(parnames):
            pars[:,i] = (self.pars[p][i1]*weights).sum(axis = 1)
            #pars[:,i] = (self.pars[p][i1]*weights1 + self.pars[p][i2]*weights2)
            if p != 'MASSIN' :
                pars[dead,i] = float('nan')
                
        stars_out = self.structure_array(pars,parnames)
        if silent is False:
            print("Padova2007: {0} of {1} initial stars have "
                  "died".format(dead.shape[0],initial_masses.shape[0]) )
    
        return stars_out

class Padova2007(Isochrone):
    hrdspydir, f = os.path.split(__file__)
    isocdir = hrdspydir + '/data/isochrones/fsps/'
    
    def __init__(self):
        self.pars = None
        flist = glob.glob(self.isocdir+'isoc*.dat')
        if len(flist) is 0 :
            raise NameError('nothing returned for ls ',isocdir,'isoc*.dat')
        self.Z_legend = [f[f.rfind('z0')+1:f.rfind('z0')+7] for f in flist]
        self.Z_list = np.array([ float(f[f.rfind('z0')+1:f.rfind('z0')+7]) for f in flist ])

    def load_all_isoc(self):
        """Globs the contents of the isochrone directory and feeds the
        metallicity list to load_isoc
        """
        self.max_mass = []
        self.logage_list = []
        
        for Z in self.Z_list:
            pars = self.load_one_isoc(Z)
            if self.pars is None:
                self.pars = pars
            else:
                self.pars = np.hstack([self.pars, pars])
            self.max_mass.append(pars['MASSIN'].max())
            
        self.logage_list = np.sort(np.unique(self.pars['LOGAGE']))
                
    def load_one_isoc(self, Z): 
        """Given a metallicity (absolute, i.e. Z_sun = 0.019) read the
        corresponding isochrone data, and use it to produce a
        structured array of stellar parameters
        """
        zval = float(Z)
        filename = glob.glob(self.isocdir+'/isoc_z{0:6.4f}.dat'.format(zval))
        age, mini, mact, logl, logt, logg, composition, phase = np.loadtxt(filename[0],
                                                                           usecols=(0,1,2,3,4,5,6,7),
                                                                           unpack=True)
        #parname = ['LOGAGE','MASSIN','MASSACT','LOGL','LOGT','LOGG','Z','COMP','PHASE','Z']
        #pars = np.loadtxt(filename[0])
        #pars = np.vstack([pars,np.zeros(pars.shape[0])+zval])
        zz = np.zeros_like(age)+zval
        pars = np.vstack([age, mini, logl, logt, logg, zz, phase, mact, composition])
        pars =  self.structure_array(pars.T, ['LOGAGE','MASSIN','LOGL',
                                              'LOGT','LOGG','Z','PHASE','MASSACT','COMP'])

        return pars

class Geneva2013(Isochrone):
    hrdspydir, f = os.path.split(__file__)
    isocdir = hrdspydir + '/data/isochrones/geneva/'
    
    def __init__(self, Vini=0.0):
        """
        :param Vini:
            Initial rotation speed, in fraction of V_breakup.  Valid
            values are 0.00 and 0.400
        """
        self.pars = None
        self.Vini = Vini
        flist = glob.glob(self.isocdir+'Isoc*Vini{0:3.2f}*.dat'.format(Vini))
        if len(flist) is 0 :
            raise NameError('nothing returned for ls {0}Isoc*Vini{1:3.2f}*.dat'.format(self.isocdir, Vini))
        Z_legend = [f[f.rfind('Z0')+1:f.rfind('Z0')+6] for f in flist]
        self.Z_list = np.unique(np.array([ float(z) for z in Z_legend]))
        self.Z_legend = ['{0:4.3f}'.format(z) for z in self.Z_list]
        logage_legend = [f[f.rfind('_t')+2:f.rfind('_t')+8] for f in flist]
        self.logage_list = np.unique(np.array([ float(t) for t in logage_legend]))
        self.logage_legend = ['{0:06.3f}'.format(t) for t in self.logage_list]
        
        
    def load_all_isoc(self):
        self.max_mass = []
        
        for Z in self.Z_list:
            for t in self.logage_list:
                pars = self.load_one_isoc(Z, t)
                if pars is None:
                    continue
                if self.pars is None:
                    self.pars = pars
                else:
                    self.pars = np.hstack([self.pars, pars])
                self.max_mass.append(pars['MASSIN'].max())

    def load_one_isoc(self, Z, logage):
        """M_ini Z_ini OmOc_ini M logL logTe_c logTe_nc MBol MV U-B
        B-V B2-V1 r_pol oblat g_pol Omega_S v_eq v_crit1 v_crit2
        Om/Om_cr lg(Md) lg(Md_M) Ga_Ed H1 He4 C12 C13 N14 O16 O17 O18
        Ne20 Ne22 Al26
        """
        Z = float(Z)

        fn = 'Isochr_Z{0:4.3f}_Vini{1:3.2f}_t{2:06.3f}.dat'.format(Z, self.Vini, logage)
        filename = glob.glob(self.isocdir + fn)
        if len(filename) == 0:
            print('Could not find {0}'.format(self.isocdir + fn))
            return None
        mini, zini, mact, logl, logt, logt_nc, mbol, rpol, logg = np.loadtxt(filename[0], skiprows=1,
                                                                    usecols=(0,1,3,4,5,6,7,12,14),
                                                                    unpack=True)
        tt = np.zeros_like(mini) + logage
        pars = np.vstack([tt, mini, logl, logt, logg, zini, mact])
        pars =  self.structure_array(pars.T, ['LOGAGE','MASSIN','LOGL',
                                              'LOGT','LOGG','Z','MASSACT'])
        
        return pars

class MIST(Isochrone):

    def __init__(self):
        """The MIST isochrones.
        """
        self.Zlist = None
        self.Zlegend = None
        self.Vini = Vini

class Parsec(Padova2007):
    """Parsec isochrones.
    """
    hrdspydir, f = os.path.split(__file__)
    isocdir = hrdspydir + '/data/isochrones/parsec/'
    
    def __init__(self, OV=0.5):
        """
        :param OV:
            Convective core overshooting.  Valid values are 0.3, 0.5, 0.7.
        """
        self.pars = None
        self.OV = OV
        flist = glob.glob(self.isocdir+'isoc*OV{0:2.1f}*.dat'.format(OV))
        if len(flist) is 0 :
            raise NameError('nothing returned for ls '
                            '{0}isoc*OV{1:2.1f}*.dat'.format(self.isocdir, Vini))
        Z_legend = [f[f.rfind('z0')+1:f.rfind('z0')+6] for f in flist]
        self.Z_list = np.unique(np.array([ float(z) for z in Z_legend]))
        self.Z_legend = ['{0:4.3f}'.format(z) for z in self.Z_list]

    def load_one_isoc(self, Z): 
        """Given a metallicity (absolute, i.e. Z_sun = 0.019) read the
        corresponding isochrone data, and use it to produce a
        structured array of stellar parameters.
        """

        #Z	log(age/yr)	M_ini   	M_act	logL/Lo	logTe	logG	mbol    F200LP1 F218W1  F225W1  F275W1  F336W1  F350LP1 F390W1  F438W1  F475W1  F555W1  F600LP1 F606W1  F625W1  F775W1  F814W1  F850LP1	C/O	M_hec	period	pmode	logMdot	slope	int_IMF	stage
        Z = float(Z)
        fn = '{0}isoc_z{1:4.3f}_OV{2:2.1f}.dat'.format(self.isocdir, Z, self.OV)
        filename = glob.glob(fn)
        zz, age, mini, mact, logl, logt, logg, mbol, composition, phase = np.loadtxt(filename[0],
                                                                           usecols=(0,1,2,3,4,5,6,7, 24, 31),
                                                                           unpack=True)
        pars = np.vstack([age, mini, logl, logt, logg, zz, phase, mact, composition])
        pars =  self.structure_array(pars.T, ['LOGAGE','MASSIN','LOGL',
                                              'LOGT','LOGG','Z','PHASE','MASSACT','COMP'])

        return pars

