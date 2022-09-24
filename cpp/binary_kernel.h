#include <torch/extension.h>
#include "libpopcnt.h"
#define MASK(a) ( (a) + ( -(a) & -((0)>(a)) ) )
#define ENCODE_BIT 16
static inline uint32_t popcnt32(uint32_t x)
{
  __asm__ ("popcnt %1, %0" : "=r" (x) : "0" (x));
  return x;
}
inline uint32_t encode_val(float* array, int remain_bit,int n ,int is_filter_transpose) {
    uint32_t sign, r = 0;
//    std::cout << "encode_val" << '\n';

    for(int i=0; i<ENCODE_BIT && i<remain_bit; i++){
        if (is_filter_transpose){
            r |= (array[i]>0)<<i;
        }
        else{
            r |= (array[i*n]>0)<<i;
        }
    }
    return r;
}

void encode_rows_cpu_kernel(float* columns, int* columns_binary, int k, int n,int is_filter_transpose) {
//Chuyển float columns sang mảng binary ở columns_binary
    int i, l = 1+(k-1)/ENCODE_BIT;
//    std::cout << "encode_rows_cpu_kernel" << '\n';

    //#pragma omp parallel for
#pragma omp parallel for private(i)
    for (i = 0; i < n*l; i++) {
//        int start_bit = l_idx*ENCODE_BITS;
        int l_idx = i%l;
        int n_idx = i/l;
        int start_bit = l_idx*ENCODE_BIT;
        int remain_bit = k - start_bit;
        int start_idx_array = 0;
        if (is_filter_transpose){
            start_idx_array = n_idx*k + start_bit;
        }
        else{
            start_idx_array = n_idx + start_bit*n;
        }
        columns_binary[i] = encode_val(&columns[start_idx_array], remain_bit,n,is_filter_transpose);
//        std::cout << columns_binary[i] << '\n';
    }
}

torch::Tensor encode_rows_cpu(torch::Tensor input,int is_filter_transpose) {
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
    encode_rows_cpu_kernel(a, b, k, n,is_filter_transpose);

//    std::cout << "encode_rows_cpu output :  " << output << '\n';
    return output;
}

torch::Tensor binary_gemm_cpu(torch::Tensor a, torch::Tensor b, int c_out, int k, int n, int beta, int alpha,int l){
//Tính a = a*b
//Compute C <- beta*C + A*B, beta = 0 or 1
//    if (c->nDimension != 2 || c->size[0]*c->size[1] < m*k) {
//        THFloatTensor_resize2d(c, m, k);
//    }
//    auto A = a.data_ptr<int>();
//    auto B = b.data_ptr<int>();
    torch::Tensor c = torch::zeros(torch::IntArrayRef({c_out,n}),torch::TensorOptions().dtype(torch::kInt32));
//    torch::Tensor alphas = torch::ones(torch::IntArrayRef({c_out,n}),torch::TensorOptions().dtype(torch::kFloat));
//    auto C = c.data_ptr<float>();
    std::cout<< c.size(0)<<"   "<<c.size(1) <<"\n";
       int n_local = 0;
  // column index
        int l_idx = 0;
        int c_channel =  0;
        auto C = c.data_ptr<int>();
    #pragma omp parallel for private(n_local)
    for (n_local=0;n_local<n;n_local++){
        #pragma omp parallel for private(c_channel)
        for ( c_channel=0;c_channel<c_out;c_channel++){
            for (l_idx=0;l_idx<l;l_idx++){
                int filter_idx = a[c_channel][l_idx].item().to<int>();
                int data_idx = b[n_local][l_idx].item().to<int>();
//                c[c_channel][n_local] += popcnt32( MASK(filter_idx ^ data_idx))<<1;
//                std::cout << filter_idx << "   " << data_idx << '\n';
//                C[n_local*n + c_channel] += popcnt32( MASK(filter_idx ^ data_idx))<<1;
                C[n_local + n*c_channel] += popcnt32(filter_idx ^ data_idx);
//                C[n_local*c_out + c_channel] += popcnt32(filter_idx ^ data_idx);
//                C[n_local + n*c_channel] += popcnt32(filter_idx ^ data_idx);
//                C[n_local*c_out + c_channel]  = 1;
            }
        }
    }
    #pragma omp parallel for
    for (n_local=0;n_local<n;n_local++){
        #pragma omp parallel for
        for ( c_channel=0;c_channel<c_out;c_channel++){

                C[n_local + n*c_channel] = k - 2*C[n_local + n*c_channel];
//                C[n_local*c_out + c_channel] = k - 2*C[n_local*c_out + c_channel];
//                C[n_local*c_out + c_channel]  = 1;
        }
    }
//    dgemm_nn(c_out, n, k, A, l, 1, B, l, 1, C, 1, k, beta, alpha, D); // wrong
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
