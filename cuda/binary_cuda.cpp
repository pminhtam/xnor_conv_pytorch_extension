#include <torch/extension.h>
//#include <torch/types.h>
#include <vector>
//#include <iostream>
//#include <cuda.h>
//#include <cuda_runtime.h>
#include "binop_cuda_kernel.cu"

torch::Tensor binary_conv2d(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
    );
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



torch::Tensor binary_conv2d(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
    ) {
    CHECK_INPUT(weights);
    CHECK_INPUT(input);
    CHECK_INPUT(bias);

    const int batch_size = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);
    const int c_out = weights.size(0), c_in = weights.size(1), k1 = weights.size(2), k2 = weights.size(3);
//    auto X = 2*input;
    torch::Tensor col_mat = torch::im2col(input,/*kernel_size=*/torch::IntArrayRef({3, 3}),
                                                 /*dilation=*/torch::IntArrayRef({1, 1}),
                                                 /*padding=*/torch::IntArrayRef({0, 0}),
                                                 /*stride=*/torch::IntArrayRef({1, 1}));

    torch::Tensor fil_2 = weights.reshape(torch::IntArrayRef({c_out, c * k1 * k2}) );  // (128, 576)

    torch::Tensor bin_fil = fil_2.clone();  // shape: (128, 576)
    torch::Tensor bin_col = col_mat.transpose(1, 2).clone();  // shape: (batch_size, 1048576, 576)

    int n = bin_col.size(1);
    int k = bin_col.size(2);
    int l = 1+(k-1)/32; // ENCODE_BITS 32
    int idx;
    torch::Tensor col_pack = torch::zeros(torch::IntArrayRef({batch_size,n,l}),torch::TensorOptions()
                    .dtype(torch::kInt32).device(torch::kCUDA, 0));

//    for(idx = 0; idx < batch_size; idx++){
//    torch::Tensor col_pack = encode_rows_cpu(bin_col[0]);
//        col_pack[idx] = encode_rows(bin_col[idx]);
//    }
    return  col_pack;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binary_conv2d_cuda", &binary_conv2d, "Binary forward conv2d cuda ");
}