hrdspy
======

generate CMDs/HRDs and integrated spectra of single age stellar populations
for arbitrary IMFs and total masses.  Also, fit stellar SEDs


installation & setup:
	1) download code to some directory 'hdir' 
	2) add setenv hrdspy = 'hdir' to your .bashrc 
	3) cp the FSPS Padova2007 isochrones to 'hdir'/data/isochrones 
	4) download the BaSeL_3.1 *.cor files of stellar spectra to the 
	  'hdir'/data/spectra/ directory and run the provided 'fitsify' 
	  idl script.
	5) download/install the observate module from sedpy, and copy any desired filters from k_correct

The basic object class is a 'Cluster', which is composed of an IMF, an isochrone library, and a stellar spectral library.  The fundamental parameters of this object are logage, total_mass, and Z.  Observational filters and dust may also be specified. The output is a library of stellar SEDs and properties, and an integrated spectrum, which are attributes of the Cluster object.

In the demo directory there is an example script example_spdisp.py which computes the integrated spectra of ten ~5e3 M_sun clusters and shows the dispersion of these spectra over the UV/optical range, as well as the average spectrum, compared to the spectrum of one 5e4 solar mass cluster.  an example of generating a CMD is also presented.

There is also a stellar SED fitter (modeled on the pyxydust fitter), for which an example is available in the demo directory

