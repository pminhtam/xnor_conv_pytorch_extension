#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define BLOCK_SIZE 16
#define BLOCK_DIM 16
#define CUDA_NUM_THREADS 1024
#define ENCODE_BITS 32


int GET_BLOCKS(int N){
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__device__ __forceinline__ uint32_t encode_val(float* array, int n) {
    uint32_t r = 0;
    for(int i=0; i<ENCODE_BITS && i<n; i++){
        r |= (array[i]>0)<<i;
    }
    return r;
}
//  Chuyển float32 -> bit

template <typename scalar_t>
__global__ void encode_rows_kernel(float *input, int* output, int m, int n, int l) {// l = 1+(n-1)/ENCODE_BITS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = n*(i/l)+ENCODE_BITS*(i%l);
    if (i<m*l) output[i] = encode_val(&input[p], n-ENCODE_BITS*(i%l));
}


// torch::Tensor encode_rows(torch::Tensor input) {
//     //THCUNN_assertSameGPU(state, 2, input, output);
//
// // chuyển float32 sang binary
//     int n = input.size(0);  // = h*w
//     int k = input.size(1);  // = c*k1*k2
//     int l = 1+(k-1)/ENCODE_BITS;
//
//     torch::Tensor output = torch::zeros(torch::IntArrayRef({n,l}),torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));
//
//     auto a = input.data_ptr<float>();
//     auto b = output.data_ptr<int>();
//
// //    encode_rows_cuda(a, b, n, k, l);
// //   const int threads = CUDA_NUM_THREADS;
// //   const dim3 blocks((state_size + threads - 1) / threads, batch_size);
//
//     AT_DISPATCH_FLOATING_TYPES(input.type(), "encode_rows",
//     ([&] {
//     encode_rows_kernel<scalar_t><<<GET_BLOCKS(n*l), CUDA_NUM_THREADS>>>(a, b, n, k, l);
//     }));
//    return output;
// }

// void encode_rows_cuda(float* input, int* output, int m, int n, int l) {
// //     encode_rows_kernel <<< GET_BLOCKS(m*l), CUDA_NUM_THREADS, 0, stream >>>(input, output, m, n, l);
//     AT_DISPATCH_FLOATING_TYPES
// }




