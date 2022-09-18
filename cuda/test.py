import torch
import binary_cuda
# h = torch.randn(32,32)
# C = torch.randn(32,32)
import torch.nn.functional as F

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
tensor_fil = tensor_fil.type(torch.int)
bias = bias.type(torch.int)
# print(bias)
# re = binary_cpp.binary_conv2d(img_pad,fil,bias)
re = binary_cuda.binary_conv2d_cuda(tensor_col,tensor_fil,bias)
# re = binary_cpp.popcnt32(124)
print(re)
print(re.shape)
# tensor_fil = tensor_fil.type(torch.float)

# out = F.conv2d(torch.tensor(tensor_col), torch.tensor(tensor_fil))
# print(out)
