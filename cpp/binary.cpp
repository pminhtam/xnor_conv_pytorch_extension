#include <torch/extension.h>
//#include <torch/types.h>
//#include <vector>
//#include <iostream>
#include "libpopcnt.h"
#include "matmul.h"

inline uint32_t encode_val(float* array, int n) {
    uint32_t sign, r = 0;
//    std::cout << "encode_val" << '\n';

    for(int i=0; i<ENCODE_BIT && i<n; i++){
        sign = array[i]>0;
        r |= (sign<<i);
    }
    return r;
}

void encode_rows_cpu_kernel(float* columns, int* columns_binary, int k, int n) {
//Chuyển float columns sang mảng binary ở columns_binary
    int i, l = 1+(k-1)/ENCODE_BIT;
//    std::cout << "encode_rows_cpu_kernel" << '\n';

    //#pragma omp parallel for
#pragma omp parallel for private(i)
    for (i = 0; i < n*l; i++) {
        int p = k*(i/l)+ENCODE_BIT*(i%l);

        columns_binary[i] = encode_val(&columns[p], n-ENCODE_BIT*(i%l));
//        std::cout << columns_binary[i] << '\n';
    }
}

torch::Tensor encode_rows_cpu(torch::Tensor input) {
// chuyển float32 sang binary
    int n = input.size(0);
    int k = input.size(1);
    int l = 1+(k-1)/ENCODE_BIT;
//    std::cout<< n << '\n';
//    std::cout<< k << '\n';
//    std::cout<< l << '\n';
//    torch::Tensor output = torch::zeros(torch::IntArrayRef({n,l}));
    torch::Tensor output = torch::zeros(torch::IntArrayRef({n,l}),torch::TensorOptions().dtype(torch::kInt32));
    auto a = input.data_ptr<float>();
    auto b = output.data_ptr<int>();
//    torch::Tensor a = input;
//    torch::Tensor b = output;
    encode_rows_cpu_kernel(a, b, k, n);

//    std::cout << "encode_rows_cpu output :  " << output << '\n';
    return output;
}

void binary_gemm_cpu(torch::Tensor a, torch::Tensor b, torch::Tensor c, int m, int nn, int k, int transb, int beta, int alpha, THFloatTensor* alphas){
//Tính a = a*b
//Compute C <- beta*C + A*B, beta = 0 or 1
    if (c->nDimension != 2 || c->size[0]*c->size[1] < m*k) {
        THFloatTensor_resize2d(c, m, k);
    }
    uint32_t *A = (uint32_t*)THIntTensor_data(a);
    uint32_t *B = (uint32_t*)THIntTensor_data(b);
    float *C = THFloatTensor_data(c);
    float *D = THFloatTensor_data(alphas);
    int n = 1 + (nn-1) / ENCODE_BIT, brow = transb? 1:k, bcol = transb? n:1;
    dgemm_nn(m, k, nn, A, n, 1, B, brow, bcol, C, k, 1, beta, alpha, D);
}

static torch::Tensor Bin_SpatialConvolutionMM_updateOutput_frame(
                                                             torch::Tensor weight,
                                                             torch::Tensor bias,
                                                             torch::Tensor *bin_col,
                                                             int kW, int kH,
                                                             int dW, int dH,
                                                             int padW, int padH,
                                                             )
{
    THFloatTensor *output2d;

    output2d = THFloatTensor_newWithStorage2d(output->storage, output->storageOffset, nOutputPlane, -1, outputHeight*outputWidth, -1);
    THFloatTensor_zero(output2d);

    binary_gemm_cpu(weight, bin_col, output2d, nOutputPlane, kW*kH*nInputPlane, outputHeight*outputWidth, 0, 1, 1, alphas);
    if (bias->nDimension) {
        THFloatTensor_addmm(output2d, 1, output2d, 1, bias, ones);
    }
    THFloatTensor_free(output2d);
}


torch::Tensor binary_conv2d(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
    ) {
    const int batch_size = input.size(0), c = input.size(1), h = input.size(2), w = input.size(3);
    const int c_out = weights.size(0), c_in = weights.size(1), k1 = weights.size(2), k2 = weights.size(3);
//    auto X = 2*input;
    torch::Tensor col_mat = torch::im2col(input,/*kernel_size=*/torch::IntArrayRef({3, 3}),
                                                 /*dilation=*/torch::IntArrayRef({1, 1}),
                                                 /*padding=*/torch::IntArrayRef({0, 0}),
                                                 /*stride=*/torch::IntArrayRef({1, 1}));
//    col_mat = col_mat.reshape(batch_size,c * k1 * k2, -1);
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
//    torch::Tensor col_pack = encode_rows_cpu(bin_col[0]);
        col_pack[idx] = encode_rows_cpu(bin_col[idx]);
    }
    torch::Tensor out_tensor = torch::zeros_like(input);

#pragma omp parallel for private(idx)
    for(idx = 0; idx < batch_size; idx++){
        out_tensor[idx] = Bin_SpatialConvolutionMM_updateOutput_frame(bin_col[idx]);
    }
    return col_pack;
//    return bin_col;
//    return fil_2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binary_conv2d", &binary_conv2d, "Binary forward conv2d cpu ");
  m.def("popcnt32", &popcnt32, "Count number of bit 1 for float32 ");
}
