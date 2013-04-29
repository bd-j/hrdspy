import numpy as np

#should implement with structured arrays to hold the
#imf segment parameters

class IMF(object):
    
    def sample(self, target_mass ):
        masses=np.zeros([])
        #if target_mass is large, should iteratively guess at the number of stars
        while masses.sum() < target_mass:
            ntry = np.max([(target_mass - masses.sum())/self.average_mass,1]) #always sample at least 1 star
            masses = np.hstack([masses,self.draw_star(ntry).flatten()])
        last = np.searchsorted(np.cumsum(masses),target_mass)
        #flip a coin to see if last star should be included so that collections of clusters are not biased low
        if np.random.uniform() > 0.5:
            last = last+1
        masses = masses[:last-1]
        print('IMF.sample: Drew %i stars with total mass %f M_sun' % (len(masses), masses.sum()) )
        return masses

    def draw_star(self,nstar = 1):
        r = np.random.uniform(0,1,nstar)
        return self.mass_at_cdf(r)

    def mass_at_cdf(self,cdf_values):
        #cumulate segment numbers and pad to allow for values that are within the first segment
        cumseg = np.insert(np.cumsum(self.number_segments),0,0) / self.total_number
        #find segment at which cmf crosses the value
        ind = np.searchsorted(cumseg, cdf_values, side='left')-1
        dp = cdf_values - cumseg[ind]
        
        x = dp * (1-self.alpha[ind])/(self.norm[ind]/self.total_number) + self.mlo[ind]**(1-self.alpha[ind])
        y = dp / (self.norm[ind]/self.total_mass) + np.log(self.mlo[ind])
        
        return  np.where( self.alpha[ind] == 1, np.exp(y), np.power(x, 1./(1.-self.alpha[ind])) )

    def integrate_powerlaw(self,low, high, power, normalization=1):
        """calculate r'norm\cdot\int_{low}^{high} x^{-power} dx'"""
        return np.where( power == 1,
                         normalization * (np.log(high) -np.log(low)),
                         normalization * (high**(1-power)-low**(1-power))/(1-power) )
    
    def cmf_at_mass(self, mass):
        cmf = 0
        for lo, hi, a, m, norm in zip(self.mlo, self.mhi, self.alpha, self.mtot, self.norm):
            if hi <= mass:
                cmf += m
            elif lo < mass:
                cmf += self.integrate_powerlaw(lo,mass,a-1,norm)
        return cmf


class SalpeterIMF(IMF):

    def __init__(self):
        self.total_mass = 1.0 #set the total mass to be 1 M_sun
        self.mlo = np.array([0.1])
        self.mhi  = np.array([100.])
        self.alpha = np.array([2.35])
        
        self.getProperties()
        
    def getProperties(self):
        self.mass_segments = np.array([self.total_mass])
        self.norm = np.atleast_1d(self.total_mass /
                                 self.integrate_powerlaw(self.mlo, self.mhi, self.alpha-1, 1).sum() )
        
        self.number_segments = self.integrate_powerlaw(self.mlo, self.mhi, self.alpha, self.norm)
        self.total_number = self.number_segments.sum()
        self.average_mass = self.mass_segments/self.number_segments

                             
class UserIMF(IMF):

    def __init__(self):
        self.sort_segments()
        self.renormalizeIMF()
        pass

    def sort_segments(self):
        """user IMF must have segments that join up (i.e. mhi[i]=mlo[i+1])
        However, the code will sort the segments, and determine normalizations
        so that they are continuous.  The normalization of the first segment must
        be greater than 0"""
        pass

    def renormalizeIMF(self):
        """make sure the power laws are continuous,
        then renormalize to a total mass of 1"""
        
        for i in xrange(1,len(self.norm)):
            #update the normalization to match the previous segement at the edge
            self.norm[i] = (self.norm[i-1] * ((self.mlo[i-1]**self.alpha[i-1]) /
                                                (self.mlo[i]**self.alpha[i])) )
            self.mass_segment[i] = self.integrate_powerlaw(self.mlo[i],self.mhi[i],self.alpha[i],self.norm[i])
        mtot = self.mass_segment.sum()
        self.norm *= self.total_mass/mtot
        self.mass_segment *= self.total_mass/mtot
