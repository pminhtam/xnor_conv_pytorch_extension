import numpy as np
from PIL import Image
import tensorflow as tf
import cv2, math
from imageio import imsave
from scipy import signal
from matplotlib import pyplot as plt
from numba import njit, cuda, vectorize, guvectorize, stencil
from numba import prange
import math


@njit
def im2col_2d(mat, fil):
    '''
    Expects input and kernel to be square shape
    Returns : im2col view with shape - (ker_sz,ker_sz,img_sz,img_sz)
    '''
    # Parameters
    c, h, w = mat.shape

    # ker_sz=fil.shape[-1]
    out_c, in_c, ker_sz, _ = fil.shape
    row_range = col_range = h - ker_sz + 1
    s0, s1, s2 = mat.strides
    shp = c, ker_sz, ker_sz, row_range, col_range
    strd = s0, s1, s2, s1, s2
    out_view = np.lib.stride_tricks.as_strided(mat, shape=shp, strides=strd)

    return out_view


def bitpack(bin_fil,bin_col,num_repeat_,num_append_):
  assert bin_fil.shape[-1] == bin_col.shape[-1]
  # Bitpack filter
  c_out , dim_k = bin_fil.shape
  num_col , _ = bin_col.shape
  a=bin_fil.repeat(num_repeat_,axis=1)  # (8, 36)
  filter_pad=np.zeros((c_out,num_append_),dtype=np.uint8)
  a_16 = np.concatenate([a,filter_pad],axis=1) # (8, 64)
  fil_pack=np.packbits(a_16).view(np.uint16).byteswap()
  fil_pack = fil_pack.reshape(a_16.shape[0],-1)     # one row can convert to multi number so have to convert
  # Bitpack image
  if num_repeat_ > 1:
    num_col_app_ = dim_k - num_col % (dim_k)

    row_pad=np.zeros((num_col_app_,dim_k),dtype=np.uint8)
    b=np.concatenate([bin_col,row_pad],axis=0) # pad col end(1048576 to 1048579)
  else:
    b = bin_col
  b=b.reshape((-1,dim_k * num_repeat_))
  col_pad=np.zeros((b.shape[0],num_append_),dtype=np.uint8)
  b=np.concatenate([b,col_pad],axis=1) # pad row end(36 to 64)
  col_pack=np.packbits(b).view(np.uint16).byteswap()
  col_pack = col_pack.reshape(b.shape[0],-1) # one row can convert to multi number so have to convert
  return fil_pack, col_pack
# Classic C-Style bit-count for unpacking
@njit
def bit_count(n):
    """Return the number of bits set to 1 in the integer number 'n'.
       This is called the Hamming weight or the population count of 'n'.
       The algorithm performs as many iterations as there are set bits.
       Argument 'n' must be non-negative'
    """
    count = 0
    while n:
        n &= n - np.uint16(1)
        count += 1
    return count


# @njit(parallel=True)
@njit
def unpack(z, bits, length_row):
    '''input: int64
      output: float array of length 7 (7*9 packs)
    '''
    c_out, wh, num_bits = z.shape
    fout = np.zeros((c_out, wh), dtype=np.float32)
    # print(num_bits)
    # print(len(bits))
    assert num_bits == len(bits)
    # print('xxxxx')
    for i in range(0, c_out):
        for k in range(wh):
            popcount = 0
            for j in range(0, num_bits):
                popcount += bit_count(z[i][k][j] & bits[j])
            fout[i][k] = 2 * popcount - length_row

    return fout

def xnor_bitwise_np(img_pad,fil):
    col_mat = im2col_2d(img_pad, fil)
    print(col_mat.shape)
    c, k1, k2, h, w = col_mat.shape
    assert c == fil.shape[1]
    assert k1 == fil.shape[2]
    assert k2 == fil.shape[3]
    c_out = fil.shape[0]
    col_mat = col_mat.reshape(c * k1 * k2, -1)  # (576, 1048576)
    fil_2 = fil.reshape(-1, c * k1 * k2)  # (128, 576)
    bin_col = np.uint8(col_mat > 0)
    bin_fil = np.uint8(fil_2 > 0)
    bin_fil = bin_fil.copy()  # shape: (128, 576)
    bin_col = bin_col.T.copy()  # shape: (1048576, 576)
    num_row_elements = c*k1*k2
    if num_row_elements < 16 :
        num_repeat_ = 16 // (c * k1 * k2)
        num_append_ = 16 % (c * k1 * k2)
    else:
        num_repeat_ = 1
        num_append_ = (16 - num_row_elements % 16)%16

    fil_pack,col_pack=bitpack(bin_fil,bin_col,num_repeat_,num_append_)
    zz = np.bitwise_xor(np.expand_dims(fil_pack, 1), np.expand_dims(col_pack, 0))
    z = np.bitwise_not(zz)
    bits = np.packbits(np.ones(num_row_elements, dtype=np.uint8)).view(np.uint16).byteswap()
    out = unpack(z, bits, num_row_elements)
    out = out.reshape(c_out, h, w)
    return out

# print(out)
import torch
import torch.nn.functional as F

def test_pytorch(img_pad,fil):

    tensor_col[img_pad > 0] = 1
    tensor_col[img_pad <= 0] = -1
    tensor_fil[fil > 0] = 1
    tensor_fil[fil <= 0] = -1
    out = F.conv2d(torch.tensor(tensor_col), torch.tensor(tensor_fil))
    return out
import time


# Initialize inputs and filters randomly
img_pad=np.random.uniform(low=-1.0, high=1.0, size=(32,1026,1026)).astype(np.float32)
fil=np.random.uniform(low=-1.0, high=1.0, size=(64,32,3,3)).astype(np.float32)


start = time.time()
out = xnor_bitwise_np(img_pad,fil)
end = time.time()
print ("Took %f ms" % ((end - start) * 1000.0))

tensor_col = torch.tensor([img_pad])
tensor_fil = torch.tensor(fil)
start = time.time()
out_torch = test_pytorch(tensor_col,tensor_fil)
end = time.time()
print ("Took %f ms" % ((end - start) * 1000.0))
print(np.sum(out- out_torch.detach().numpy()))
# import timeit
#
# print(timeit.timeit('xnor_bitwise_np(img_pad,fil)'))
# print(timeit.timeit('test_pytorch(img_pad,fil)'))
