#simple exploration of the effects of the cmf on
#inferred SFHs

import numpy as np
from imf import IMF
#threshold = #number of clusters per unit mass considered to be 'dangerous'

class CMF(IMF):

    def __init__(self):
        self.total_mass = 1.0 #set the total mass to be 1 M_sun
        self.mlo = np.array([1e2])
        self.mhi  = np.array([1e5])
        self.alpha = np.array([2.0])

        self.update_properties()
        
    def update_properties(self):
        self.mass_segments = np.array([self.total_mass])
        self.norm = np.atleast_1d(self.total_mass /
                                 self.integrate_powerlaw(self.mlo, self.mhi, self.alpha-1, 1).sum() )
        
        self.number_segments = self.integrate_powerlaw(self.mlo, self.mhi, self.alpha, self.norm)
        self.total_number = self.number_segments.sum()
        self.average_mass = self.mass_segments/self.number_segments

    def sfh(self, sfr = 1.0, interval = 1e8, timebins = None):
        mtot = sfr * interval
        masses = self.sample(mtot)
        ages = np.random.uniform(0, interval, len(masses))
        if timebins is not None:
            sfh, edges = np.histogram(ages, weights= masses, bins = timebins)
            sfh = sfh/np.diff(edges)
        else:
            sfh = None
        return masses, ages, sfh
    
