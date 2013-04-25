import os
import glob
import numpy as np
import utils


class Padova2007(object):
    isocdir = os.getenv('SPS_HOME')+'ISOCHRONES/Padova/Padova2007/'

    def __init__(self):
        self.isoc = None
        pass

    def load_all_isoc(self):
        flist = glob.glob(self.isocdir+'isoc*.dat')
        if len(flist) is 0 : raise NameError('nothing returned for ls ',isocdir,'isoc*.dat')
        self.Z_list=[]
        self.Z_legend = []
        for i,f in enumerate(flist):
            Z=f[f.rfind('z0')+1:f.rfind('z0')+7]
            self.Z_legend.append(Z)
            self.Z_list.append(float(Z))
            self.load_one_isoc(Z)
        self.Z_list=np.array(self.Z_list)
        
    def load_one_isoc(self, Z):
        if type(Z) is float : Z = '%6.4f' % Z
        zval=float(Z)
        filename = glob.glob(self.isocdir+'/isoc*'+Z+'*.dat')
        age, mini, mact, logl, logt, logg, composition, phase = np.loadtxt(filename[0],usecols=(0,1,2,3,4,5,6,7),unpack=True)
        #parname = ['LOGAGE','MASSIN','MASSACT','LOGL','LOGT','LOGG','Z','COMP','PHASE','Z']
        #pars = np.loadtxt()
        #pars = np.vstack([pars,np.zeros(pars.shape[0])+zval])
        
        zz = np.zeros_like(age)+zval
        pars = np.vstack([age, mini, logl, logt, logg, zz, phase, mact, composition])
        pars = utils.structure_array(pars.T, parname = ['LOGAGE','MASSIN','LOGL',
                                                        'LOGT','LOGG','Z','PHASE','MASSACT','COMP'])
        if self.isoc is None:
            self.isoc = pars
        else:
            self.isoc = np.hstack([self.isoc, pars])

        self.logage_list=np.sort(np.unique(self.isoc['LOGAGE']))
        
    def stellar_properties(self,initial_masses,logage,Z = 0.019,silent = False):
        #could interpolate with Delaunay Triangulation, but using faster dumb nearest neighbor for now
        this_age = self.logage_list[utils.nearest_index(self.logage_list,logage)]
        this_Z = self.Z_list[utils.nearest_index(self.Z_list, Z)]
        if silent is False:
            print('Padova2007: looking for stars with log(age) = %f and Z = %f' %(this_age,this_Z))
        #print(this_age, this_age.shape, this_Z, this_Z.shape)
        inds = np.where(np.logical_and(self.isoc['LOGAGE'] == this_age, self.isoc['Z'] == this_Z))
        inds = inds[0]
        
        #now interpolate in (log) mass, keeping weights
        #print(self.isoc['MASSIN'].shape, np.unique(self.isoc['MASSIN'][inds]).shape)
        order = self.isoc['MASSIN'][inds].argsort()
        mini = self.isoc['MASSIN'][inds[order]]
        ind_nearest = np.searchsorted( mini, initial_masses,side='left')
        #clip dead stars.  
        dead = np.where(ind_nearest == mini.shape[0])[0]
        ind_nearest = np.clip(ind_nearest,0,mini.shape[0]-1)
        i1, i2 = inds[order[ind_nearest]], inds[order[ind_nearest-1]]
        
        d1 = np.log(self.isoc['MASSIN'][i1]) - np.log(initial_masses)
        d2 = np.log(self.isoc['MASSIN'][i2]) - np.log(initial_masses)
        weights1, weights2 = np.abs(d1)/np.absolute(d1-d2), np.abs(d2)/np.absolute(d1-d2)

        #should loop this over a parameter name list with arrayToStruct to assemble output
        parnames = self.isoc.dtype.names
        pars = np.zeros([weights1.shape[0],len(parnames)])
        for i,p in enumerate(parnames):
            pars[:,i] = (self.isoc[p][i1]*weights1 + self.isoc[p][i2]*weights2)
        pars[dead,:] = float('nan')
        stars_out = utils.structure_array(pars,parnames)
        if silent is False:
            print('Padova2007: %i of %i input stars have died' %(dead.shape[0],initial_masses.shape[0]) )
    
        return stars_out
