#include <torch/extension.h>
#include "binary_kernel.h"

#pragma omp parallel
torch::Tensor binary_conv2d(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
    ) {
    const int batch_size = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);
    const int c_out = weights.size(0), c_in = weights.size(1), k1 = weights.size(2), k2 = weights.size(3);
    torch::Tensor col_mat = torch::im2col(input,/*kernel_size=*/torch::IntArrayRef({3, 3}),
                                                 /*dilation=*/torch::IntArrayRef({1, 1}),
                                                 /*padding=*/torch::IntArrayRef({0, 0}),
                                                 /*stride=*/torch::IntArrayRef({1, 1}));
    torch::Tensor fil_2 = weights.reshape(torch::IntArrayRef({c_out, c * k1 * k2}) );  // (128, 576)

    torch::Tensor bin_fil = fil_2.clone();  // shape: (128, 576)
    torch::Tensor bin_col = col_mat.transpose(1, 2).clone();  // shape: (batch_size, 1048576, 576)

    int n = bin_col.size(1);
    int k = bin_col.size(2);
    int l = 1+(k-1)/ENCODE_BIT;
    int idx;
    torch::Tensor col_pack = torch::zeros(torch::IntArrayRef({batch_size,n,l}),torch::TensorOptions().dtype(torch::kInt32));
#pragma omp parallel for private(idx)
    for(idx = 0; idx < batch_size; idx++){
        col_pack[idx] = encode_rows_cpu(bin_col[idx],0);
    }
    torch::Tensor fil_pack = torch::zeros(torch::IntArrayRef({c_out, l}),torch::TensorOptions()
                    .dtype(torch::kInt32));
    fil_pack = encode_rows_cpu(bin_fil,1);
    torch::Tensor out_tensor = torch::zeros(torch::IntArrayRef({batch_size,c_out,n}));

#pragma omp parallel for private(idx)
    for(idx = 0; idx < batch_size; idx++){
        out_tensor[idx] = Bin_SpatialConvolutionMM_updateOutput_frame(fil_pack,bias,col_pack[idx],
         c_in, k1,k2,n, c_out,l);
    }
    return out_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binary_conv2d", &binary_conv2d, "Binary forward conv2d cpu ");
  m.def("popcnt32", &popcnt32, "Count number of bit 1 for float32 ");
}
