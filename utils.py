import numpy as np

def nearest_index(array, value):
    return (np.abs(array-value)).argmin()

def structure_array(values,fieldnames, types=['<f8']):
    """turn a numpy array of floats into a structurd array. fieldnames can be a list or
    string array of parameter names with length nfield.
    Assumes pars is a numpy array of shape (nobj,nfield)
    """
    values=np.atleast_2d(values)
    if values.shape[-1] != len(fieldnames):
        if values.shape[0] == len(fieldnames):
            values=values.T
        else:
            raise ValueError('models.ModelGrid.arrayToStruct: array and fieldnames do not have consistent shapes!')
    nobj=values.shape[0]
        
    #set up the list of tuples describing the fields.  Assume each parameter is a float unless otherwise specified
    fieldtuple=[]
    for i,f in enumerate(fieldnames):
        if len(types) > 1 :
            tt =types[i]
        else: tt=types[0]
        fieldtuple.append((f,tt))
    #create the dtype and structured array                    
    dt=np.dtype(fieldtuple)
    struct=np.empty(nobj,dtype=dt)
    for i,f in enumerate(fieldnames):
        struct[f]=values[...,i]
    return struct

def join_struct_arrays(arrays):
    """from some dudes on StackOverflow.  add equal length
    structured arrays to produce a single structure with fields
    from both.  input is a sequence of arrays."""
    if False in [len(a) == len(arrays[0]) for a in arrays] :
        raise ValueError ('join_struct_arrays: array lengths do not match.')

    newdtype = sum((a.dtype.descr for a in arrays), [])
    if len(np.unique(newdtype.names)) != len(newdtype.names):
        raise ValueError ('join_struct_arrays: arrays have duplicate fields.')
    newrecarray = np.empty(len(arrays[0]), dtype = newdtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray


    
def weightsDT(model_points,target_points):
    """ The interpolation weights are determined from barycenter coordinates
    of the vertices of the enclosing Delaunay triangulation. This allows for
    the use of irregular Nd grids. see also weightsLinear.
    model_points - array of shape (nmod, ndim)
    target_points - array of shape (ntarg,ndim)
    output inds and weights - arrays of shape (npts,ndim+1)
    """
        
    ndim = target_points.shape[-1]
    #delaunay triangulate and find the encompassing (hyper)triangle(s) for the desired point
    dtri = scipy.spatial.Delaunay(model_points)
    #output triangle_inds is an (ntarg) array of simplex indices
    triangle_inds = dtri.find_simplex(target_points)
    #and get model indices of (hyper)triangle vertices. inds has shape (ntarg,ndim+1)
    inds = dtri.vertices[triangle_inds,:]
    #get the barycenter coordinates through matrix multiplication and dimensional juggling
    bary = np.dot( dtri.transform[triangle_inds,:ndim,:ndim],
                   (target_points-dtri.transform[triangle_inds,ndim,:]).reshape(-1,ndim,1) )
    oned = np.arange(triangle_inds.shape[0])
    bary = np.atleast_2d(np.squeeze(bary[oned,:,oned,:])) #ok.  in np 1.7 can add an axis to squeeze
    last = 1-bary.sum(axis=-1) #the last bary coordinate is 1-sum of the other coordinates
    weights = np.hstack((bary,last[:,np.newaxis]))

    return inds, weights 
