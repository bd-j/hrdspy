import numpy as np
from numpy import squeeze, argsort, log10, radians, degrees, sin, cos, arctan2, hypot, tan


#######
### function to obtain likelihood ####
#######

def lnprob_grid(grid, obs, err, mask):
    #linearize fluxes.  
    mod_maggies = 10**(0-grid.sed/2.5)
    obs_maggies = 10**(0-obs/2.5)
    obs_ivar = (obs_maggies*err/1.086)**(-2)

    #best scale for these parameters.  sums are over the bands
    #lstar = np.squeeze(( (mod_maggies*obs_maggies*obs_ivar).sum(axis=-1) ) /
    #                   ( ( (mod_maggies**2.0)*obs_ivar ).sum(axis=-1) ))
    #lbol = grid.lbol*lstar
    #probability with dimensional juggling to get the broadcasting right
    #chi2 =  (( (lstar*mod_maggies.T).T - obs_maggies)**2)*obs_ivar
    #delta_mag = 0-2.5*np.log10((lstar*mod_maggies.T).T/obs_maggies)

    chi2 = (( mod_maggies - obs_maggies)**2)*obs_ivar
    lnprob = squeeze(-0.5*chi2[:,mask].sum(axis=-1))
    delta_mag = 0 - 2.5*log10(mod_maggies/obs_maggies)
    #clean NaNs
    
    return lnprob, delta_mag

def sortgrid(pars, sortlist = None):
    if sortlist is None:
        sortlist = pars.dtype.names
    parorder = {}
    for p in sortlist:
        parorder[p] = argsort(pars[p])
    return parorder


#not working yet....
def cdf_moment(par,lnprob,percentiles,save=False,plot=False):
    """Generate CDF and return percentiles of it."""
    order = np.argsort(par)
    cdf = np.cumsum(np.exp(lnprob[order])) / np.sum(np.exp(lnprob))
    ind_ptiles= np.searchsorted(cdf,percentiles)
    ind_max=np.argmax(lnprob_isnum)

    return np.concatenate(par[order[ind_ptiles]],par[ind_max])

def prob_less_than(par, lnprob, values):
    """Determine the CDF at a particular parameter value."""
    pass

def spheredist(ra1, dec1, ra2, dec2):
    """Returns great circle distance (and position angle).  Inputs in degrees.

    Uses vicenty distance formula - a bit slower than others, but
    numerically stable.  From E. Tollerud, with position angle calculation added."""

    from numpy import radians, degrees, sin, cos, arctan2, hypot, tan

    # terminology from the Vicenty formula - lambda and phi and
    # "standpoint" and "forepoint"
    lambs = radians(ra1)
    phis = radians(dec1)
    lambf = radians(ra2)
    phif = radians(dec2)

    dlamb = lambf - lambs

    numera = cos(phif) * sin(dlamb)
    numerb = cos(phis) * sin(phif) - sin(phis) * cos(phif) * cos(dlamb)
    numer = hypot(numera, numerb)
    denom = sin(phis) * sin(phif) + cos(phis) * cos(phif) * cos(dlamb)

    theta  = arctan2(sin(dlamb), cos(phis) * tan(phif) - sin(phis) * cos(dlamb))
    
    return degrees(arctan2(numer, denom)), degrees(theta)


def structure_array(values,fieldnames, types=['<f8']):
    """turn a numpy array of floats into a structurd array. fieldnames can be a list or
    string array of parameter names with length nfield.
    Assumes pars is a numpy array of shape (nobj,nfield)
    """
    values = np.atleast_2d(values)
    if values.shape[-1] != len(fieldnames):
        if values.shape[0] == len(fieldnames):
            values = values.T
        else:
            raise ValueError('modelgrid.structure_array: array and fieldnames do not have consistent shapes!')
    nobj = values.shape[0]
        
    #set up the list of tuples describing the fields.  Assume each parameter is a float
    fieldtuple = []
    for i,f in enumerate(fieldnames):
        if len(types) > 1 :
            tt =types[i]
        else: tt=types[0]
        fieldtuple.append((f,tt))
        #create the dtype and structured array                    
    dt = np.dtype(fieldtuple)
    struct = np.zeros(nobj,dtype=dt)
    for i,f in enumerate(fieldnames):
        struct[f] = values[...,i]
    return struct

def join_struct_arrays(arrays):
    """from some dudes on StackOverflow.  add equal length
    structured arrays to produce a single structure with fields
    from both.  input is a sequence of arrays."""
    if False in [len(a) == len(arrays[0]) for a in arrays] :
        raise ValueError ('join_struct_arrays: array lengths do not match.')
    
    newdtype = np.dtype(sum((a.dtype.descr for a in arrays), []))        
    if len(np.unique(newdtype.names)) != len(newdtype.names):
        raise ValueError ('join_struct_arrays: arrays have duplicate fields.')
    newrecarray = np.empty(len(arrays[0]), dtype = newdtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray
