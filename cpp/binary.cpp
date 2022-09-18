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
    int n = input.size(0);  // = h*w
    int k = input.size(1);  // = c*k1*k2
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

torch::Tensor binary_gemm_cpu(torch::Tensor a, torch::Tensor b, int c_out, int k, int n, int beta, int alpha,int l){
//Tính a = a*b
//Compute C <- beta*C + A*B, beta = 0 or 1
//    if (c->nDimension != 2 || c->size[0]*c->size[1] < m*k) {
//        THFloatTensor_resize2d(c, m, k);
//    }
    auto A = a.data_ptr<int>();
    auto B = b.data_ptr<int>();
    torch::Tensor c = torch::zeros(torch::IntArrayRef({c_out,n}),torch::TensorOptions().dtype(torch::kFloat));
    torch::Tensor alphas = torch::ones(torch::IntArrayRef({c_out,n}),torch::TensorOptions().dtype(torch::kFloat));
    auto C = c.data_ptr<float>();
    auto D = alphas.data_ptr<float>();
//    int l = 1 + (k-1) / ENCODE_BIT, brow = transb? 1:n, bcol = transb? l:1;
//    dgemm_nn(c_out, n, k, A, l, 1, B, brow,s bcol, C, k, 1, beta, alpha, D);
//    int a_s1 =  a.size(0);
//    std::cout << a_s1<< '\n';
//    std::cout << a.size(0)<< '\n';
//    std::cout << a.size(1)<< '\n';
//    dgemm_nn(c_out, n, k, A, l, 1, B, 1, l, C, k, 1, beta, alpha, D);   // wrong big
//    dgemm_nn(c_out, n, k, A, l, 1, B, l, 1, C, k, 1, beta, alpha, D); // wrong big
//    dgemm_nn(c_out, n, k, A, 1, l, B, 1, l, C, 1, k, beta, alpha, D); // wrong
//    dgemm_nn(c_out, n, k, A, 1, l, B, l, 1, C, 1, k, beta, alpha, D); // wrong
    dgemm_nn(c_out, n, k, A, l, 1, B, l, 1, C, 1, k, beta, alpha, D); // wrong
    // ??????????
    return c;

}

static torch::Tensor Bin_SpatialConvolutionMM_updateOutput_frame(
                                                             torch::Tensor weight,
                                                             torch::Tensor bias,
                                                             torch::Tensor bin_col,
                                                             int c_in,int k1, int k2,
                                                             int n, int c_out,int l
                                                             )
{


    torch::Tensor output2d = binary_gemm_cpu(weight, bin_col, c_out, c_in*k1*k2, n, 1, 1,l);
//    if (bias->nDimension) {
//        THFloatTensor_addmm(output2d, 1, output2d, 1, bias, ones);
//    }
//    THFloatTensor_free(output2d);
    return output2d;
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
//    torch::Tensor out_tensor = torch::zeros_like(input);
    torch::Tensor out_tensor = torch::zeros(torch::IntArrayRef({batch_size,c_out,n}));

#pragma omp parallel for private(idx)
    for(idx = 0; idx < batch_size; idx++){
        out_tensor[idx] = Bin_SpatialConvolutionMM_updateOutput_frame(weights,bias,col_pack[idx],
         c_in, k1,k2,n, c_out,l);
    }
//    return col_pack;
    return out_tensor;
//    return bin_col;
//    return fil_2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binary_conv2d", &binary_conv2d, "Binary forward conv2d cpu ");
  m.def("popcnt32", &popcnt32, "Count number of bit 1 for float32 ");
}
