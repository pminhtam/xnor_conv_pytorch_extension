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


__device__ __forceinline__ uint32_t encode_val(float* array, int n) {
    uint32_t r = 0;
    // float r2 = 0;
    for(int i=0; i<ENCODE_BITS && i<n; i++){
        r |= (array[i]>0)<<i;
    }
    // r2 = r;
    return r;
}
//  Chuyển float32 -> bit

template <typename scalar_t>
__global__ void encode_rows_kernel(float *input, int* output, int m, int n, int l) {// l = 1+(n-1)/ENCODE_BITS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = n*(i/l)+ENCODE_BITS*(i%l);
    if (i<m*l) output[i] = encode_val(&input[p], n-ENCODE_BITS*(i%l));
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
  // const int threads = 1024;
  // const dim3 blocks(16,16);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "encode_rows", ([&] {
    encode_rows_kernel<scalar_t><<<GET_BLOCKS(n*l), CUDA_NUM_THREADS>>>(
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
__global__ void binary_gemm_kernel(int* A, int* B, float* C, int m, int nn, int k, int transb, int alpha, int beta) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

	int n = 1 + (nn-1)/ENCODE_BITS;
    int startLocation = BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol;

    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

    int Cvalue = 0;

    int c = blockIdx.x*blockDim.x + threadIdx.x;
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int lim = 1+( (n-1) / BLOCK_SIZE);
    for (int i = 0; i < lim; ++i) {

        // Get sub-matrix Asub of A
        int* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];

        // Get sub-matrix Bsub of B
        int* Bsub = transb? &B[BLOCK_SIZE * blockCol * n + BLOCK_SIZE * i] : &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];

        if ((BLOCK_SIZE*i+col)<n && r<m)
            As[row][col] = Asub[row*n+col];
        else
            As[row][col] = 0;
        if ((BLOCK_SIZE*i+row)<n && c<k)
            Bs[row][col] = transb? Bsub[row+col*n] : Bsub[row*k+col];
        else
            Bs[row][col] = 0;

        __syncthreads();
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; ++j)
            Cvalue += __popc(As[row][j]^Bs[j][col]);
//             Hàm xnor ở đây
        __syncthreads();
    }
// hiệu chỉnh lại giá trị xnor vì bit là [0,1] còn bnn là [-1,1 ]
// Chỉ chọn vùng có giá trị, loại vùng thừa
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m){
		Csub[row*k+col] = beta ? Csub[row*k+col]:0;
// 		Csub[row*k+col]+= alpha? (1.0*nn-(Cvalue<<1))*alphas[(startLocation+row*k+col)/k] : 1.0*nn-(Cvalue<<1);
		Csub[row*k+col]+= alpha? (1.0*nn-(Cvalue<<1)) : 1.0*nn-(Cvalue<<1);
	}
}

torch::Tensor binary_gemm(torch::Tensor a, torch::Tensor b,  int c_out, int k, int n, int transb, int alpha, int beta){

    torch::Tensor c = torch::zeros(torch::IntArrayRef({n,c_out}),torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0));
//     b : columns_binary
    auto A = a.data_ptr<int>();
    auto B = b.data_ptr<int>();
    auto C = c.data_ptr<float>();

//     float *D = alpha? THCudaTensor_data(state, alphas) : NULL;
//     cudaStream_t stream = THCState_getCurrentStream(state);

//     binary_gemm_cuda(A, B, C, c_out, k, n, transb, alpha, beta);
    dim3 blockDim(ENCODE_BITS, ENCODE_BITS, 1);
    dim3 gridDim(k/ENCODE_BITS+1, n/ENCODE_BITS+1, 1);

    AT_DISPATCH_FLOATING_TYPES(c.type(), "binary_gemm", ([&] {
    binary_gemm_kernel<scalar_t><<<gridDim, blockDim>>>(
     A, B, C, c_out, k, n, 0, 1, 1
    );
    }));
//    return output;
    return c.transpose(0, 1);
}