import numpy as np
import astropy.io.fits as pyfits
from StringIO import StringIO
from astropy.io import ascii

import observate
import utils

##########
#### Data Inputs ####
#########

def load_image_cube(rp):
    """Convert input catalogs into 1 x nstar x nfilter arrays for the fitter"""

    #read optical catalogs and optionally the matched galex catalog
    if rp['source'] is 'mcps':
        optical, lines = read_some_mcps(**rp)
        header = utils.join_struct_arrays([optical[['RAh','Dec','flag']],
                                           utils.structure_array(lines,['line_number'])])
    elif rp['source'] is 'massey':
        optical = read_massey(**rp)
        header = optical[['RAh','Dec','spType']]
        if 'galex_NUV' in rp['fnamelist']:
            galex = match_galex_to_opt_byID(optical, rp['galex_csv_filename'],
                                          match_type = 'massey_id',
                                          criterion = 'distance_arcsec', ind = 0 )
            header = utils.join_struct_arrays([header,
                                               galex[['NUV_exptime','fov_radius','distance_arcsec']]])

    #set up arrays to hold the matched SEDs
    nstar = len(optical)
    data_mag = np.zeros( [1, nstar, len(rp['fnamelist'])] )
    data_magerr = np.zeros( [1, nstar, len(rp['fnamelist'])] )
    
    #fill the arrays with magnitude information, including ab to vega corrctions
    for i,fname in enumerate(rp['fnamelist']):
        filt = observate.Filter(fname)
        if fname in optical.dtype.names:
            data_mag[...,i] = optical[fname]-filt.ab_to_vega
            data_magerr[...,i] = optical[fname+'_unc']
        elif fname in galex.dtype.names:
            data_mag[...,i] = galex[fname]
            data_magerr[...,i] = galex[fname+'_unc']
    return data_mag, data_magerr, header

def read_some_mcps(catalog_name = 'table1.dat', lines = None, nlines = 100, fsize = 24.1e6, **extras):
    """Read in just a selection of lines from the full MCPS catalog"""

    bytes_per_line = 74 #for the MCPS LMC table1.dat
    dt = {'names':('RAh','Dec',
                   'bessell_U','bessell_U_unc',
                   'bessell_B','bessell_B_unc',
                   'bessell_V','bessell_V_unc',
                   'bessell_I','bessell_I_unc',
                   'flag'),
          'formats':('<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<i4')}
    if lines is None:
        print('randomly choosing {0} lines'.format(nlines))
        lines = np.sort(np.random.uniform(0, fsize, nlines).astype(int))

    #read specific line numbers from the catalog and store in a StringIO object
    f = open(catalog_name, 'r')
    somelines = ''
    for lno in lines:
        f.seek(lno * bytes_per_line)
        somelines+=f.readline()
    f.close()
    somelines = StringIO(somelines)

    #turn into a numpy structured array
    data = np.loadtxt(somelines, dtype = dt)
    return data, lines

def read_massey(catalog_name = None, spec_types = True, spec_catalog = None, **extras):
    """Read in data from Massey 2002, by default only include stars with spectral types."""

    table = ascii.read(catalog_name)
    if spec_types is True: #restrict to stars spectrally typed
        sptable = ascii.read(spec_catalog)
        table = table[sptable['CNum']-1]  

    dt = {'names':('RAh','Dec',
                   'bessell_U','bessell_U_unc',
                   'bessell_B','bessell_B_unc',
                   'bessell_V','bessell_V_unc',
                   'bessell_R','bessell_R_unc',
                   'spType', 'r_spType'),
          'formats':('<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','a12','a8')}
    newt = np.zeros(len(table), dtype = dt)

    newt['RAh'] = (table['RAh'] + table['RAm']/60. + table['RAs']/3600.)
    newt['Dec'] = (-1)*(table['DEd'] + table['DEm']/60. + table['DEs']/3600.)
    newt['bessell_V'] = table['Vmag']
    newt['bessell_B'] = table['Vmag'] + table['B-V']
    newt['bessell_U'] = table['Vmag'] + table['B-V'] + table['U-B']
    newt['bessell_R'] = table['Vmag'] - table['V-R']
    newt['bessell_V_unc'] = table['e_Vmag']
    newt['bessell_B_unc'] = np.sqrt(table['e_Vmag']**2 + table['e_B-V']**2) #use np.hypot
    newt['bessell_U_unc'] = np.sqrt(table['e_Vmag']**2 + table['e_B-V']**2 + table['e_U-B']**2)
    newt['bessell_R_unc'] = np.sqrt(table['e_Vmag']**2 + table['e_V-R']**2)

    if spec_types is True:
        newt['spType'] = sptable['SpType']
        newt['r_spType'] = sptable['r_SpType']
    else:
        newt['spType'][sptable['CNum']-1] = sptable['SpType']
        newt['r_spType'][sptable['CNum']-1] = sptable['r_SpType']

    return newt


def read_galex_csv(catalog_name = None):
    """Read in the output from GALEXView, return a normal numpy structured array
    alternatively, could rename columns in the astropy table...."""

    gdata = ascii.read(catalog_name, fill_values = [('---','-999'), ('--','-99')])
    matched = ((gdata['distance_arcmin'] > 0) & (gdata['distance_arcmin'] < (12/60.)))
    gdata = gdata[matched] #restrict to galaxies with matches and reasonable offsets
    dt = {'names':('RAh','Dec',
                   'galex_FUV','galex_FUV_unc',
                   'galex_NUV','galex_NUV_unc',
                   'FUV_exptime','NUV_exptime',
                   'distance_arcsec','fov_radius',
                   'nuv_artifact','massey_id',
                   'survey', 'tilename'),
            'formats':('<f8','<f8','<f8','<f8','<f8','<f8',
                       '<f8','<f8','<f8','<f8','<i8','<i8','a4','a14')
            }

    galex = np.zeros(len(gdata), dtype = dt)
    galex['RAh'] = gdata['ra']/15.
    galex['Dec'] = gdata['dec']
    galex['galex_FUV'] = gdata['fuv_mag']
    galex['galex_FUV_unc'] = np.sqrt(gdata['fuv_magerr']**2+0.04**2)
    galex['galex_NUV'] = gdata['nuv_mag']
    galex['galex_NUV_unc'] = np.sqrt(gdata['nuv_magerr']**2+0.04**2)
    galex['FUV_exptime'] = gdata['fuv_exptime']
    galex['NUV_exptime'] = gdata['nuv_exptime']
    galex['distance_arcsec'] = gdata['distance_arcmin']*60.
    galex['fov_radius'] = gdata['fov_radius']
    galex['nuv_artifact'] = gdata['nuv_artifact']
    galex['massey_id'] = gdata['uploadID']-1
    galex['survey'] = gdata['survey']
    galex['tilename'] = gdata['tilename']
    return galex

def match_galex_to_opt_byID(optical, galex_csv_name, match_type = 'massey_id',
                            criterion = 'distance_arcsec', ind = 0 ):
    
    """Match a GALEXView based catalog to a set of stars, using
    the criterion parameter to decide between multiple matches.
    Use ind = 0 for a minmumum of criterion, ind = -1 for maximum"""

    allg = read_galex_csv(catalog_name = galex_csv_name)
    galex = np.zeros(len(optical), dtype = allg.dtype)
    for i in xrange(len(optical)):
        this = ((allg[match_type] == i) & 
                (allg['fov_radius'] < 0.55) &
                (allg['distance_arcsec'] < 7)
                )
            #print(match_type, this.sum())
        if this.sum() == 1:
            #print(i)
            galex[i] = allg[this]
        elif this.sum() > 1:
            order = np.argsort(allg[criterion][this])
            galex[i] = (allg[this])[order[ind]]
    return galex

def write_for_galex(cat, outfile, idroot = ''):
    """Write a table suitable for the GALEXView search tool"""

    out = open(outfile,'wb')
    out.write('ID,RA,DEC\n')
    for i in xrange(len(cat)):
        out.write(idroot+'{0:05d}, {1}, {2}\n'.format(i+1, cat[i]['RAh']*15., cat[i]['Dec']))
    out.close()
