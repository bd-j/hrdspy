import os, glob
import numpy as np
from sedpy.modelgrid import *

#import utils
#import sedmodels

class Padova2007(ModelLibrary):
    hrdspydir, f = os.path.split(__file__)
    isocdir = hrdspydir + '/data/isochrones/'

    def __init__(self):
        self.pars = None

    def load_all_isoc(self):
        """Globs the contents of the isochrone directory and feeds the
        metallicity list to load_isoc
        """
        flist = glob.glob(self.isocdir+'isoc*.dat')
        if len(flist) is 0 :
            raise NameError('nothing returned for ls ',isocdir,'isoc*.dat')
        self.Z_list = []
        self.Z_legend = [f[f.rfind('z0')+1:f.rfind('z0')+7] for f in flist]
        self.Z_list = [float(f[f.rfind('z0')+1:f.rfind('z0')+7]) for f in flist]
        self.load_isoc(self.Z_legend)
        self.Z_list = np.array(self.Z_list)

    def load_isoc(self, Zlist):
        """Reads each isochrone in sequence to produce a single
        structured array of all stellar parameters, and store those in
        the pars attribute.
        """
        self.max_mass = []
        self.logage_list = []
        for i,Z in enumerate(Zlist):
            pars = self.load_one_isoc(Z)
            if self.pars is None:
                self.pars = pars
            else:
                self.pars = np.hstack([self.pars, pars])
            
            self.max_mass.append(pars['MASSIN'].max())
        self.logage_list = np.sort(np.unique(pars['LOGAGE']))
        
    def load_one_isoc(self, Z): 
        """Given a metallicity (absolute, i.e. Z_sun = 0.019) read the
        corresponding isochrone data, and use it to produce a
        structured array of stellar parameters
        """
        if type(Z) is float :
            Z = '%6.4f' % Z
        zval = float(Z)
        filename = glob.glob(self.isocdir+'/isoc*'+Z+'*.dat')
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
    
    def get_stellar_pars_at(self,initial_masses,logage,Z = 0.019,silent = False):
        """Given an array of initial masses, an age (in log yrs) and a
        metallicity, interpolate the isochrone stellar parameter
        values to these masses, age, and metallicity.  Interpolation
        in age and metallicity is nearest neighbor. Interpolation in
        mass uses peicewise linear weights in log(mass).
        """
        # Could interpolate with Delaunay Triangulation, but using
        # faster and safer nearest neighbor for now
        this_age = self.logage_list[self.nearest_index(self.logage_list,logage)]
        this_Z = self.Z_list[self.nearest_index(self.Z_list, Z)]
        if silent is False:
            print("Padova2007: looking for stars with "
                  "log(age) = {0} and Z = {1}".format(this_age,this_Z))
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
