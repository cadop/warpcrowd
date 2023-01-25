

def quad_to_tri(a):
    idx = np.flatnonzero(a[:,-1] == 0)
    out0 = np.empty((a.shape[0],2,3),dtype=a.dtype)      

    out0[:,0,1:] = a[:,1:-1]
    out0[:,1,1:] = a[:,2:]

    out0[...,0] = a[:,0,None]

    out0.shape = (-1,3)

    mask = np.ones(out0.shape[0],dtype=bool)
    mask[idx*2+1] = 0
    return out0[mask]

    