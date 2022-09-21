#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define BLOCK_SIZE 16
#define BLOCK_DIM 16
#define CUDA_NUM_THREADS 1024
#define ENCODE_BITS 16


int GET_BLOCKS(int N){
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__device__ __forceinline__ int encode_val(float* array, int remain_bit,int n) {
    int r = 0;
    // float r2 = 0;
    for(int i=0; i<ENCODE_BITS && i<remain_bit; i++){
        r |= (array[i*n]>0)<<i;
    }
    // r2 = r;
    return r;
}
//  Chuyển float32 -> bit

template <typename scalar_t>
__global__ void encode_rows_kernel(float *input, int* output, int n, int k, int l) {// l = 1+(n-1)/ENCODE_BITS
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int l_idx = blockIdx.y;
    int start_bit = l_idx*ENCODE_BITS;
    int remain_bit = k - start_bit;
//     int p = n*(i/l)+ENCODE_BITS*(i%l);
    if (n_idx<n & l_idx<l & start_bit<k & remain_bit>0) {
        output[l_idx + n_idx*l] = encode_val(&input[n_idx + start_bit*n ], remain_bit,n);
//         output[l_idx + n_idx*l] = encode_val(&input[n_idx + start_bit*n_idx ], remain_bit,n);
    }
}

torch::Tensor encode_rows(torch::Tensor input) {
    //THCUNN_assertSameGPU(state, 2, input, output);

// chuyển float32 sang binary
    int n = input.size(0);  // = h*w
    int k = input.size(1);  // = c*k1*k2
    int l = 1+(k-1)/ENCODE_BITS;

    torch::Tensor output = torch::zeros(torch::IntArrayRef({n,l}),torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));
    // torch::Tensor output = torch::zeros(torch::IntArrayRef({n,l}));

    auto a = input.data_ptr<float>();
    auto b = output.data_ptr<int>();

//    encode_rows_cuda(a, b, n, k, l);
//   const int threads = CUDA_NUM_THREADS;
//   const dim3 blocks((state_size + threads - 1) / threads, batch_size);
  const int threads = 1024;
//   const dim3 threads(1024,);
  const dim3 blocks(n/1024 + 1 ,l);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "encode_rows", ([&] {
    encode_rows_kernel<scalar_t><<<blocks, threads>>>(
     a, b, n, k, l
    );
    }));
   return output;
}

// void encode_rows_cuda(float* input, int* output, int m, int n, int l) {
// //     encode_rows_kernel <<< GET_BLOCKS(m*l), CUDA_NUM_THREADS, 0, stream >>>(input, output, m, n, l);
//     AT_DISPATCH_FLOATING_TYPES
// }
template <typename scalar_t>
__global__ void binary_gemm_kernel(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> a,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b,
        torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> c,
        int c_out , int l,int k, int n) {
        // ta se tính theo  b (tức data ) n =
        // a : filter : = (c_out, num_int): = (c_out, c_in*k1*k2)
        // b : data : = (n , num_int) = (n, c_in*k1*k2)
        // c output := (c_out, n)
  //batch index
  const int n_local = threadIdx.x + blockIdx.x * blockDim.x;
  // column index
  const int idx_int_num_local = threadIdx.y;
  const int c_channel =  blockIdx.y;
    int idx = 0;
  if (n_local < n & idx_int_num_local < l & c_channel < c_out){
    //  b[idx_int_num_local][n_local] = 1;
      // auto b_element = b[n_local][idx_int_num_local];
    // c[c_channel][idx_int_num_local][n_local] = 1;
// c[c_channel][idx_int_num_local][n_local] = b[0][0];
// c[c_channel][idx_int_num_local][n_local] = b[n_local][idx_int_num_local];
c[c_channel][idx_int_num_local][n_local] = 2*__popc( (unsigned int) a[c_channel][idx_int_num_local]^ (unsigned int) b[n_local][idx_int_num_local])-k;
// c[c_channel][idx_int_num_local][n_local] = __popc(12344);

// auto rere = __popc(a[c_channel][idx_int_num_local]^b[n_local][idx_int_num_local]);

    // for (idx = 0; idx<c_out; idx++){
    //   const auto b_element = b[n_local][idx_int_num_local];
    //     // c[idx][idx_int_num_local][n_local] = __popc(a[idx][idx_int_num_local]^b[n_local][idx_int_num_local]);
    //       //  auto CC = __popc(a[idx][idx_int_num_local]^b[n_local][idx_int_num_local]);
    //       // auto zz = a[idx][idx_int_num_local];
    //       // auto zz = b[n_local][idx_int_num_local];
    //       // a[idx][idx_int_num_local] = 1;
    //       // b[idx_int_num_local][n_local] = 1;
    //       // c[idx][idx_int_num_local][n_local] = 1;
    //         // c[idx][idx_int_num_local][n_local] = a[idx][idx_int_num_local]+b[n_local][idx_int_num_local];
    //         // c[idx][idx_int_num_local][n_local] = a[idx][idx_int_num_local];
    //         // c[idx][idx_int_num_local][n_local] = b[n_local][idx_int_num_local];
    //         c[idx][idx_int_num_local][n_local] = b_element;
    // }
  }
}

torch::Tensor binary_gemm(torch::Tensor a, torch::Tensor b,  int c_out, int l, int k, int n, int transb, int alpha, int beta){

//     b : columns_binary
//     auto A = a.data_ptr<int>();
//     auto B = b.data_ptr<int>();
//     auto C = c.data_ptr<float>();

//     float *D = alpha? THCudaTensor_data(state, alphas) : NULL;s
//     cudaStream_t stream = THCState_getCurrentStream(state);
//  k số hàng
// n số cột
//     binary_gemm_cuda(A, B, C, c_out, k, n, transb, alpha, beta);
//     dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
//     dim3 gridDim(k/ENCODE_BITS+1, n/ENCODE_BITS+1, 1);
    // const int kk = 1+(k-1)/ENCODE_BITS;
    // const int threads = 1024;
    const dim3 threads(256, l/1+1);
    const dim3 blocks(n/256+1, c_out);
    torch::Tensor c = torch::zeros(torch::IntArrayRef({c_out,l,n}),torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));
    // std::cout << b.size(0) << "  ,   " << b.size(1) << '\n';
    // std::cout << "b[0][0] : " << b[0][0] << '\n';
    // std::cout << "a[0][0] : " << a[0][0] << '\n';
    AT_DISPATCH_ALL_TYPES(b.type(), "binary_gemm", ([&] {
    binary_gemm_kernel<scalar_t><<<blocks, threads>>>(
     a.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
     b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
     c.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    c_out ,l,k, n
    );
    }));
//    return output;
//     return c.transpose(0, 1);
    return c;
}