import numpy as np
from numba import njit
import torch



@njit
def im2col_2d(mat, fil, res=1):
    '''
    Expects input and kernel to be square shape
    Returns : im2col view with shape - (ker_sz,ker_sz,img_sz,img_sz)
    '''
    # Parameters
    row_range = col_range = len(mat) - len(fil) + 1
    ker_sz = len(fil)
    s0, s1 = mat.strides
    shp = ker_sz, ker_sz, row_range, col_range
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(mat, shape=shp, strides=strd)

    return out_view

# Initialize inputs and filters randomly
img_pad=np.random.uniform(low=-1.0, high=1.0, size=(1026,1026)).astype(np.float32)
fil=np.random.uniform(low=-1.0, high=1.0, size=(3,3)).astype(np.float32)

col_mat=im2col_2d(img_pad, fil,res=1)
col_mat = col_mat.reshape(len(col_mat)*len(col_mat),-1) #(9, 1048576)

# im2col reshape
col_mat=im2col_2d(img_pad, fil,res=1)
col_mat = col_mat.reshape(len(col_mat)*len(col_mat),-1) #(9, 1048576)
fil_mat=fil.reshape(1,len(fil)*len(fil)) # (1, 9)

# Binarize col_mat and fil_mat
bin_col=np.uint8(col_mat>0)
bin_fil=np.uint8(fil_mat>0)

# calculate xnor between col and fil
out_xnor=np.logical_not(np.logical_xor(bin_fil.reshape((-1,1)),bin_col))
print("Out shape: {}, Out type: {}".format(out_xnor.shape, out_xnor.dtype))
out = 2*(np.sum(out_xnor,axis=0)) - len(out_xnor)

result = out.reshape((1024,1024))

print(result.shape)