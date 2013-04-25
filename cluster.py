import imf
import isochrone

class Cluster(object):
    def __init__(self,target_mass, logage, Z, isoc = None, IMF = None, speclib = None)
        self.target_mass = target_mass
        self.logage = logage
        self. Z = Z
        self.compose(isoc, IMF, speclib)
        
    def compose(self, IMF, isoc, speclib):
        if IMF is None:
            self.imf  = imf.SalpeterIMF()
        else:
            self.imf = IMF
        if isoc is None:
            self.evol=isochrone.Padova2007()
            self.evol.load_all_isoc()
        else:
            self.evol = isoc
        if speclib is None:
            self.speclib = sedmodels.BaSeL()
            self.speclib.load_all_spectra()
        else:
            self.speclib = speclib
            
    def generate_stars(self, self.loagage, self.Z):
        star_masses = self.imf.sample(self.total_mass)
        self.total_mass_formed = star_masses.sum()
        
        self.stars = self.evol.stellar_properties(star_masses, logage, Z )
        self.nstars=self.stars.shape[0]

    def observe_stars(self, stars,filterlist):
        self.spectra = self.speclib.generateSpectra(stars)
        mags = observate.getSED(basel.wavelength, cluster.spectra, filterlist )


