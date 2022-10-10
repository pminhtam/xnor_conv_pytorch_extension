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


__device__ __forceinline__ int encode_val(float* array, int remain_bit,int n,int is_filter_transpose) {
    int r = 0;
    for(int i=0; i<ENCODE_BITS && i<remain_bit; i++){
        if (is_filter_transpose){
            r |= (array[i]>0)<<i;
        }
        else{
            r |= (array[i*n]>0)<<i;
        }
    }
    return r;
}
//  Chuyển float32 -> bit

template <typename scalar_t>
__global__ void encode_rows_kernel(float *input, int* output, int n, int k, int l,int is_filter_transpose) {// l = 1+(n-1)/ENCODE_BITS
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int l_idx = blockIdx.y;
    int start_bit = l_idx*ENCODE_BITS;
    int remain_bit = k - start_bit;
    if (n_idx<n & l_idx<l & start_bit<k & remain_bit>0) {
        if (is_filter_transpose) {
            output[l_idx + n_idx*l] = encode_val(&input[start_bit + n_idx*k ], remain_bit,n,is_filter_transpose);
        }
        else{
            output[l_idx + n_idx*l] = encode_val(&input[n_idx + start_bit*n ], remain_bit,n,is_filter_transpose);
        }
    }
}

torch::Tensor encode_rows(torch::Tensor input,int is_filter_transpose) {

// chuyển float32 sang binary
    int n = input.size(0);  // = h*w
    int k = input.size(1);  // = c*k1*k2
    int l = 1+(k-1)/ENCODE_BITS;

    torch::Tensor output = torch::zeros(torch::IntArrayRef({n,l}),torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));

    auto a = input.data_ptr<float>();
    auto b = output.data_ptr<int>();


  const int threads = 1024;
  const dim3 blocks(n/1024 + 1 ,l);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "encode_rows", ([&] {
    encode_rows_kernel<scalar_t><<<blocks, threads>>>(
     a, b, n, k, l,is_filter_transpose
    );
    }));
   return output;
}


template <typename scalar_t>
__global__ void binary_gemm_kernel(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> a,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> b,
        torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> c,
        int c_out , int l,int k, int n) {
  //batch index
  const int n_local = threadIdx.x + blockIdx.x * blockDim.x;
  // column index
  const int idx_int_num_local = threadIdx.y;
  const int c_channel =  blockIdx.y;
  if (n_local < n & idx_int_num_local < l & c_channel < c_out){
///
// Đây là hàm xor, để tính xnor = k-nor;
c[c_channel][idx_int_num_local][n_local] = __popc( (unsigned int) a[c_channel][idx_int_num_local]^ (unsigned int) b[n_local][idx_int_num_local]);
  }
}

template <typename scalar_t>
__global__ void get_final_result_kernel(
            const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output_bin,
            torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_bin_final,
            int c_out, int n, int l,int k) {// l = 1+(n-1)/ENCODE_BITS
  //batch index
  const int n_local = threadIdx.x + blockIdx.x * blockDim.x;
  // column index
  const int c_channel =  blockIdx.y;
    int idx_l = 0;
  if (n_local < n & c_channel < c_out){
        for (idx_l = 0; idx_l <l;idx_l++){
            output_bin_final[c_channel][n_local] += output_bin[c_channel][idx_l][n_local];
        }
        output_bin_final[c_channel][n_local] = k-2*output_bin_final[c_channel][n_local];
    }
}

torch::Tensor binary_gemm(torch::Tensor a, torch::Tensor b,  int c_out, int l, int k, int n, int transb, int alpha, int beta){


//  k số hàng
// n số cột
//     int num_thread = int(1024/(l+1);
    const dim3 threads((int) 1024/(l+1) , l/1+1);
    const dim3 blocks(n/256+1, c_out);
    torch::Tensor c = torch::zeros(torch::IntArrayRef({c_out,l,n}),torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));
    AT_DISPATCH_ALL_TYPES(b.type(), "binary_gemm binary_gemm_kernel", ([&] {
    binary_gemm_kernel<scalar_t><<<blocks, threads>>>(
     a.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
     b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
     c.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
    c_out ,l,k, n
    );
    }));

    const dim3 threads2(1024);
    const dim3 blocks2(n/1024+1, c_out);
    torch::Tensor d = torch::zeros(torch::IntArrayRef({c_out,n}),torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));
    AT_DISPATCH_ALL_TYPES(b.type(), "binary_gemm get_final_result_kernel", ([&] {
    get_final_result_kernel<scalar_t><<<blocks2, threads2>>>(
     c.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
     d.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
    c_out ,n, l,k
    );
    }));
    return d;
}