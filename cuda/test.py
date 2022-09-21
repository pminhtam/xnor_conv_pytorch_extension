import torch
import binary_cuda
# h = torch.randn(32,32)
# C = torch.randn(32,32)
import torch.nn.functional as F

import numpy as np
torch.manual_seed(0)
from py.xnor_bitwise_numpy import im2col_2d, bitpack
def xnor_bitwise_np(img_pad,fil):
    img_pad = img_pad.cpu().numpy()
    fil = fil.cpu().numpy()
    img_pad = img_pad[0]
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
    print(bin_col[0])
    print(bin_col[1])
    print(bin_col[2])
    print(bin_col[3])
    print(bin_col[4])
    print(bin_col[6])
    print(bin_col[7])
    print(bin_col[8])
    print(bin_col[9])
    print(bin_col[10])
    print(bin_col[11])
    fil_pack,col_pack=bitpack(bin_fil,bin_col,num_repeat_,num_append_)
    return fil_pack,col_pack


# img_pad = torch.randn(3,32,1026,1026)   # batch , c, h,w,
img_pad = torch.randn(1,3,1026,1026,device='cuda')   # batch , c, h,w,
# fil = torch.randn(64,32,3,3) # c_out, c_in, k1,k2
fil = torch.randn(64,3,3,3,device='cuda') # c_out, c_in, k1,k2
bias = torch.zeros(64,32,device='cuda') # c_out, c_in, k1,k2

tensor_col = img_pad
tensor_fil = fil

tensor_col[img_pad > 0] = 1
tensor_col[img_pad <= 0] = 0
tensor_fil[fil > 0] = 1
tensor_fil[fil <= 0] = 0
# tensor_col = tensor_col.type(torch.int)
# tensor_fil = tensor_fil.type(torch.int)
# bias = bias.type(torch.int)
# print(bias)

# re = binary_cpp.binary_conv2d(img_pad,fil,bias)
re = binary_cuda.binary_conv2d_cuda(tensor_col,tensor_fil,bias)
# re = binary_cpp.popcnt32(124)
# print(re[0][0])
# print(re[0])
print(re[0,:11,:].cpu().numpy())
print(re.shape)
print(xnor_bitwise_np(img_pad,fil)[1])

# tensor_fil = tensor_fil.type(torch.float)

# out = F.conv2d(torch.tensor(tensor_col), torch.tensor(tensor_fil))
# print(out)
