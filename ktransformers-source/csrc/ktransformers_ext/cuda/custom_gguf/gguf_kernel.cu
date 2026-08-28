// adapted from https://github.com/jinzhen-lin/vllm/blob/ea3970282137f8dcc04d3265680a3366408494fe/csrc/quantization/gguf/gguf_kernel.cu
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <vector>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"

// Q8 gemv
template <typename scalar_t>
static __global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx,
                                   const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

// Q8_K128 gemv
template <typename scalar_t>
static __global__ void quantize_q8_k128_bak(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  constexpr int kBlockSize = 128;
  
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_K128* y = (block_q8_K128*)vy;

  const int ib = i_padded / kBlockSize;   // block index
  const int iqs = i_padded % kBlockSize;  // quant index
  const int bsum_idx = iqs / 32;          // block sum index (每32个元素一个sum)

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  
  // 计算每个block内的最大绝对值
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
  }
  
  // 在warp之间同步最大值
  if (threadIdx.x % 32 == 0) {
    atomicMax((int*)&y[ib].d, __float_as_int(amax));
  }
  __syncthreads();
  
  // 获取最终的scale
  float d = __int_as_float(atomicMax((int*)&y[ib].d, __float_as_int(amax)));
  float id = (d != 0.0f) ? 127.0f / d : 0.0f;
  
  // 量化
  int8_t q = (d != 0.0f) ? roundf(xi * id) : 0;
  y[ib].qs[iqs] = q;
  
  // 计算每32个元素的和
  int sum = q;
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }
  
  // 只有每个warp的第一个线程写入sum
  if (threadIdx.x % 32 == 0) {
    y[ib].bsums[bsum_idx] = sum;
  }
  
  // 最后设置scale值
  if (iqs == 0) {
    y[ib].d = 1 / id;
  }
}

// Q8_K128 gemv
// 这个版本的实现略好一点，可能在于用过了共享内存
template <typename scalar_t>
static __global__ void quantize_q8_k128(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  constexpr int kBlockSize = 128;
  
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_K128* y = (block_q8_K128*)vy;

  const int ib = i_padded / kBlockSize;   // block index
  const int iqs = i_padded % kBlockSize;  // quant index
  const int bsum_idx = iqs / 32;          // block sum index (每32个元素一个sum)

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  
  // 计算每个block内的最大绝对值
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
  }
  
  // 在warp之间同步最大值 - 使用共享内存替代原子操作
  __shared__ float block_max[4]; // 假设每个block最多有4个warp
  __shared__ float block_d;
  if (threadIdx.x % 32 == 0) {
    block_max[threadIdx.x / 32] = amax;
  }
  __syncthreads();
  
  // 只在第一个线程计算最终的最大值
  if (threadIdx.x == 0) {
    float max_val = block_max[0];
    for (int i = 1; i < 4; i++) {
      max_val = fmaxf(max_val, block_max[i]);
    }
    block_d = max_val / 127.0f;
    y[ib].d = max_val / 127.0f;
  }
  __syncthreads();
  
  // 获取最终的scale
  float d = block_d;
  float id = (d != 0.0f) ? 1.0f / d : 0.0f;
  
  // 量化
  int8_t q = (d != 0.0f) ? roundf(xi * id) : 0;
  y[ib].qs[iqs] = q;
  // 计算每32个元素的和
  int sum = q;
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }
  
  // 只有每个warp的第一个线程写入sum
  if (threadIdx.x % 32 == 0) {
    y[ib].bsums[bsum_idx] = sum;
  }
}


template <typename scalar_t>
static void quantize_row_q8_k128_cuda(const scalar_t* x, void* vy, const int kx,
                                   const int ky, cudaStream_t stream) {
  constexpr int kBlockSize = 128;
  // 确保kx是kBlockSize的倍数
  assert(kx % kBlockSize == 0);
  
  const int64_t kx_padded = (kx + kBlockSize - 1) / kBlockSize * kBlockSize;
  const int block_num_x =
      (kx_padded + kBlockSize - 1) / kBlockSize;
  const dim3 num_blocks(block_num_x, ky, 1);
  const dim3 block_size(kBlockSize, 1, 1);
  
  // 初始化vy中的d值为0
  // cudaMemsetAsync(vy, 0, ky * (kx / kBlockSize) * sizeof(block_q8_K128), stream);
  
  quantize_q8_k128<scalar_t>
      <<<num_blocks, block_size, 0, stream>>>(x, vy, kx, kx_padded);
}

torch::Tensor ggml_dequantize(torch::Tensor W,  // quant weight
                              int64_t type, int64_t m, int64_t n) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto options =
      torch::TensorOptions().dtype(torch::kFloat16).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(type);
  to_fp16_cuda((void*)W.data_ptr(), (half*)DW.data_ptr(), m * n, stream);
  return DW;
}

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W,  // quant weight
                                  torch::Tensor X,  // input
                                  int64_t type, int64_t row) {
  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({1, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({1, padded / 32 * 9}, options);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(),
                                     (void*)quant_X.data_ptr(), col, 1, stream);
    switch (type) {
      case 2:
        mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 3:
        mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 6:
        mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 7:
        mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 8:
        mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 10:
        mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 11:
        mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 12:
        mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 13:
        mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 14:
        mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 16:
        mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 17:
        mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 18:
        mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 19:
        mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 20:
        mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 21:
        mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 22:
        mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 23:
        mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
      case 29:
        mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_mul_mat_vec_q8_k128(torch::Tensor W,  // quant weight
                                  torch::Tensor X,  // input
                                  int64_t type, int64_t row) {
  constexpr int kBlockSize = 128;
  int col = X.sizes()[1];
  const int padded = (col + kBlockSize - 1) / kBlockSize * kBlockSize;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({1, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  // (128 * 8 + 32 + 4 * 16) /32 = 32 + 1 + 2 = 35
  at::Tensor quant_X = torch::empty({1, padded / kBlockSize * 35}, options);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_vec_q8_k128", [&] {
    quantize_row_q8_k128_cuda<scalar_t>((scalar_t*)X.data_ptr(),
                                     (void*)quant_X.data_ptr(), col, 1, stream);    
    switch (type) {
      // only implement iq1_s_r4_q8_k128
      case 219:
        mul_mat_vec_iq1_s_r4_q8_k128_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_mul_mat_a8(torch::Tensor W,  // quant weight
                              torch::Tensor X,  // input
                              int64_t type, int64_t row) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  int batch = X.sizes()[0];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({batch, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_moe_a8(torch::Tensor X,  // input
                          torch::Tensor W,  // expert weights
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, tokens, stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_moe_vec_q8_k128(torch::Tensor X,  // input
                          torch::Tensor W,  // expert weights
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens) {
  // sorted_token_ids: [tokens, qlen], vec=1时无意义
  // expert_ids: [tokens,top_k]
  // num_tokens_post_padded: 始终为1
  // tokens 代表输出到底需要多少个token
  constexpr int kBlockSize = 128;
  int col = X.sizes()[1];
  int x_tokens = X.sizes()[0];
  int padded = (col + kBlockSize - 1) / kBlockSize * kBlockSize;
  assert(row % 4 == 0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({x_tokens, padded / kBlockSize * 35}, options);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe", [&] {
    quantize_row_q8_k128_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, x_tokens, stream);
    switch (type) {
      case 219:{
        const int row_size = 2 + col * 1.5 / 8;
        const int expert_stride = row_size * row;
        ggml_moe_vec_iq1_s_r4_q8_k128_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), expert_stride, col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      }
      case 229:{
        const int row_size = 2 + col *  7 / 32;
        const int expert_stride = row_size * row;
        ggml_moe_vec_iq1_m_r4_q8_k128_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), expert_stride, col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      }
    }
  });
  return Y;
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MMQ_X_Q4_0;
    case 3:
      return MMQ_X_Q4_1;
    case 6:
      return MMQ_X_Q5_0;
    case 7:
      return MMQ_X_Q5_1;
    case 8:
      return MMQ_X_Q8_0;
    case 10:
      return MMQ_X_Q2_K;
    case 11:
      return MMQ_X_Q3_K;
    case 12:
      return MMQ_X_Q4_K;
    case 13:
      return MMQ_X_Q5_K;
    case 14:
      return MMQ_X_Q6_K;
  }
  return 0;
}

torch::Tensor moe_gemm( 
  torch::Tensor X,                      // input:   [token_num,hidden_size]
  torch::Tensor W,                      // Weights: quantized and packed weighst
  torch::Tensor sorted_token_ids_all,       // [token_num], [0,topk,topk*2,...,topk*(tok_num-1)]
  torch::Tensor expert_ids,             // [token_num * max_topk]，
  torch::Tensor num_tokens_post_padded_all, // [1] = token_num * topk
  int64_t type, int64_t row,
  torch::Tensor idx_dev, 
  torch::Tensor topk_dev, 
  torch::Tensor sorted_slice_start_dev,
  torch::Tensor expert_slice_start_dev,
  torch::Tensor expert_slice_end_dev,
  int64_t max_topk,
  int64_t max_tokens
) {
  using torch::indexing::Slice;
  int x_tokens = X.sizes()[0];
  constexpr int kBlockSize = 128;
  int col = X.sizes()[1];
  int padded = (col + kBlockSize - 1) / kBlockSize * kBlockSize;
  assert(row % 4 == 0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({max_tokens, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({x_tokens, padded / kBlockSize * 35}, options);
  const auto token_num = x_tokens;
  // p1: expert_slice_start=0, top_k=flex_topk; p2: expert_slice_start=flex_topk, top_k=(max_topk-flex_topk)
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe", [&] {
    quantize_row_q8_k128_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                          col, x_tokens, stream);
    switch (type) {
      case 219:{
        const int row_size = 2 + col * 1.5 / 8;
        const int expert_stride = row_size * row;
        ggml_moe_vec_iq1_s_r4_q8_k128_cuda_dyn(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(), (scalar_t*)Y.data_ptr(),
            sorted_token_ids_all.data_ptr<int>(),
            expert_ids.data_ptr<int>(),
            num_tokens_post_padded_all.data_ptr<int>(),
            idx_dev.data_ptr<int>(),
            topk_dev.data_ptr<int>(),
            sorted_slice_start_dev.data_ptr<int>(),
            expert_slice_start_dev.data_ptr<int>(),
            expert_stride, col, row,
            (int)max_tokens, padded, row,
            (int)max_topk, (int)max_tokens, stream);
        break;
      }
      case 229:{
        const int row_size = 2 + col *  7 / 32;
        const int expert_stride = row_size * row;
        ggml_moe_vec_iq1_m_r4_q8_k128_cuda_dyn(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(), (scalar_t*)Y.data_ptr(),
            sorted_token_ids_all.data_ptr<int>(),
            expert_ids.data_ptr<int>(),
            num_tokens_post_padded_all.data_ptr<int>(),
            idx_dev.data_ptr<int>(),
            topk_dev.data_ptr<int>(),
            sorted_slice_start_dev.data_ptr<int>(),
            expert_slice_start_dev.data_ptr<int>(),
            expert_stride, col, row,
            (int)max_tokens, padded, row,
            (int)max_topk, (int)max_tokens, stream);
        break;
      }
    }
  });
  return Y;
}

torch::Tensor moe_gemm_w2( 
torch::Tensor X,                      // input:   [token_num,hidden_size]
torch::Tensor W,                      // Weights: quantized and packed weighst
torch::Tensor sorted_token_ids_all,       // [token_num], [0,topk,topk*2,...,topk*(tok_num-1)]
torch::Tensor expert_ids,             // [token_num * max_topk]，
torch::Tensor num_tokens_post_padded_all, // [1] = token_num * topk
int64_t type, int64_t row, 
int64_t top_k,
torch::Tensor idx_dev,
torch::Tensor topk_dev,
torch::Tensor sorted_slice_start_dev,
int64_t max_topk,
int64_t max_tokens
) {
  constexpr int kBlockSize = 128;
  int col = X.sizes()[1];
  int x_tokens = X.sizes()[0];
  int padded = (col + kBlockSize - 1) / kBlockSize * kBlockSize;
  assert(row % 4 == 0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({max_tokens, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({x_tokens, padded / kBlockSize * 35}, options);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe", [&] {
    quantize_row_q8_k128_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                          col, x_tokens, stream);
  switch (type) {
    case 219:{
      const int row_size = 2 + col * 1.5 / 8;
      const int expert_stride = row_size * row;
      ggml_moe_vec_iq1_s_r4_q8_k128_cuda_dyn_w2(
          (void*)quant_X.data_ptr(), (void*)W.data_ptr(), (scalar_t*)Y.data_ptr(),
          sorted_token_ids_all.data_ptr<int>(),
          expert_ids.data_ptr<int>(),
          num_tokens_post_padded_all.data_ptr<int>(),
          idx_dev.data_ptr<int>(),
          topk_dev.data_ptr<int>(), // flex_topk_dev
          sorted_slice_start_dev.data_ptr<int>(),
          expert_stride, col, row,
          (int)max_tokens, padded, row,
          (int)top_k,
          (int)max_topk, (int)max_tokens, stream);
      break;
    }
    case 229:{
      const int row_size = 2 + col *  7 / 32;
      const int expert_stride = row_size * row;
      ggml_moe_vec_iq1_m_r4_q8_k128_cuda_dyn_w2(
          (void*)quant_X.data_ptr(), (void*)W.data_ptr(), (scalar_t*)Y.data_ptr(),
          sorted_token_ids_all.data_ptr<int>(),
          expert_ids.data_ptr<int>(),
          num_tokens_post_padded_all.data_ptr<int>(),
          idx_dev.data_ptr<int>(),
          topk_dev.data_ptr<int>(), // flex_topk_dev
          sorted_slice_start_dev.data_ptr<int>(),
          expert_stride, col, row,
          (int)max_tokens, padded, row,
          (int)top_k,
          (int)max_topk, (int)max_tokens, stream);
      break;
    }
  }
});
return Y;
}

template <typename dst_t>
__device__ __forceinline__ dst_t cast_from_float(float v);

template <>
__device__ __forceinline__ half cast_from_float<half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ nv_bfloat16 cast_from_float<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename dst_t>
__global__ void dequantize_iq1_s_r4_matrix_kernel(
    const uint8_t* __restrict__ expert_ptr,
    dst_t* __restrict__ out,
    int n_per_row,
    int num_rows) {
  const int row_group = blockIdx.y;
  const int block_idx = blockIdx.x;
  const int lane = threadIdx.x;
  if (lane >= 32 || row_group >= num_rows / 4 || block_idx >= n_per_row / 32) {
    return;
  }

  const int k = lane / 8;
  const int i = (lane % 8) / 2;
  const int j = lane % 2;
  const int idx_in_row = block_idx * 32 + i * 8 + j * 4;

  const int nblock = n_per_row / 32;
  const int group_bytes = 8 + nblock * static_cast<int>(sizeof(block_iq1_s_r4));
  const uint8_t* group_ptr = expert_ptr + row_group * group_bytes;
  const half* d_scales = reinterpret_cast<const half*>(group_ptr);
  const block_iq1_s_r4* blocks = reinterpret_cast<const block_iq1_s_r4*>(group_ptr + 8);
  const half base_d = d_scales[k];
  const block_iq1_s_r4 blk = blocks[block_idx];
  const uint16_t qh = blk.qh[k];
  const int idx = static_cast<int>(blk.qs[4 * i + k]) | (((qh >> (3 * i)) & 0x07) << 8);

  uint32_t grid32[2];
  grid32[0] = iq1s_grid_gpu[idx];
  grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
  grid32[0] &= 0x0f0f0f0f;
  const int8_t* q = reinterpret_cast<const int8_t*>(grid32);

  const float d1q = __half2float(base_d) * (((qh >> 11) & 0x0E) + 1);
  const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
  const int row = row_group * 4 + k;
  out[row * n_per_row + idx_in_row + 0] = cast_from_float<dst_t>(d1q * (q[j * 4 + 0] + delta));
  out[row * n_per_row + idx_in_row + 1] = cast_from_float<dst_t>(d1q * (q[j * 4 + 1] + delta));
  out[row * n_per_row + idx_in_row + 2] = cast_from_float<dst_t>(d1q * (q[j * 4 + 2] + delta));
  out[row * n_per_row + idx_in_row + 3] = cast_from_float<dst_t>(d1q * (q[j * 4 + 3] + delta));
}

template <typename dst_t>
__global__ void dequantize_iq1_m_r4_matrix_kernel(
    const uint8_t* __restrict__ expert_ptr,
    dst_t* __restrict__ out,
    int n_per_row,
    int num_rows) {
  const int row_group = blockIdx.y;
  const int block_idx = blockIdx.x;
  const int lane = threadIdx.x;
  if (lane >= 32 || row_group >= num_rows / 4 || block_idx >= n_per_row / 32) {
    return;
  }

  // One thread corresponds to (k, j): k=row in [0..3], j=element in [0..7].
  // This mirrors llama.cpp dequantize_row_iq1_m_r4 indexing directly.
  const int k = lane / 8;      // row in {0,1,2,3}
  const int t = lane % 8;      // per-row worker id in {0..7}
  const int i = t / 4;         // chunk id in {0,1}
  const int s = t % 4;         // sub-chunk id in {0,1,2,3}
  const int half_sel = s / 2;  // 0 -> idx1, 1 -> idx2
  const int within4 = s % 2;   // first 4 or second 4 values
  const int nblock = n_per_row / 32;
  const int group_bytes = 8 + nblock * static_cast<int>(sizeof(block_iq1_m_r4));
  const uint8_t* group_ptr = expert_ptr + row_group * group_bytes;
  const half* d_scales = reinterpret_cast<const half*>(group_ptr);
  const block_iq1_m_r4* blocks = reinterpret_cast<const block_iq1_m_r4*>(group_ptr + 8);
  const block_iq1_m_r4 blk = blocks[block_idx];
  const uint8_t qh_byte = blk.qh[4 * i + k];
  const int idx1 = static_cast<int>(blk.qs[8 * i + k + 0]) | ((qh_byte & 0x07) << 8);
  const int idx2 = static_cast<int>(blk.qs[8 * i + k + 4]) | ((qh_byte & 0x70) << 4);
  const int idx = (half_sel == 0) ? idx1 : idx2;
  uint32_t grid32[2];
  grid32[0] = iq1s_grid_gpu[idx];
  grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
  grid32[0] &= 0x0f0f0f0f;
  const int8_t* q = reinterpret_cast<const int8_t*>(grid32);
  const float base_d = __half2float(d_scales[k]);
  const float dl0 = base_d * float(blk.scales[k] & 0x0f);
  const float dl1 = base_d * float(blk.scales[k] >> 4);
  const float d = (i == 0) ? dl0 : dl1;
  // Keep dequant expression consistent with vecdot_iq1_m_r4_q8_k128_shared:
  // value = d * (q + (-1 +/- IQ1M_DELTA))
  const float delta1 = (qh_byte & 0x08) ? (-1.0f - IQ1M_DELTA) : (-1.0f + IQ1M_DELTA);
  const float delta2 = (qh_byte & 0x80) ? (-1.0f - IQ1M_DELTA) : (-1.0f + IQ1M_DELTA);
  const float delta = (half_sel == 0) ? delta1 : delta2;
  const int row = row_group * 4 + k;
  const int out_base = block_idx * 32 + 16 * i + 8 * half_sel + 4 * within4;
  out[row * n_per_row + out_base + 0] = cast_from_float<dst_t>(d * (q[within4 * 4 + 0] + delta));
  out[row * n_per_row + out_base + 1] = cast_from_float<dst_t>(d * (q[within4 * 4 + 1] + delta));
  out[row * n_per_row + out_base + 2] = cast_from_float<dst_t>(d * (q[within4 * 4 + 2] + delta));
  out[row * n_per_row + out_base + 3] = cast_from_float<dst_t>(d * (q[within4 * 4 + 3] + delta));
}

static void launch_dequantize_iq1_r4(
    const uint8_t* expert_ptr,
    int64_t type,
    int col,
    int row,
    const at::Tensor& w_out,
    cudaStream_t stream) {
  const int nblock = col / 32;
  dim3 grid(nblock, row / 4, 1);
  dim3 block(32, 1, 1);

  if (w_out.scalar_type() == torch::kFloat16) {
    if (type == 219) {
      dequantize_iq1_s_r4_matrix_kernel<half><<<grid, block, 0, stream>>>(
          expert_ptr, reinterpret_cast<half*>(w_out.data_ptr()), col, row);
    } else {
      dequantize_iq1_m_r4_matrix_kernel<half><<<grid, block, 0, stream>>>(
          expert_ptr, reinterpret_cast<half*>(w_out.data_ptr()), col, row);
    }
  } else if (w_out.scalar_type() == torch::kBFloat16) {
    if (type == 219) {
      dequantize_iq1_s_r4_matrix_kernel<nv_bfloat16><<<grid, block, 0, stream>>>(
          expert_ptr, reinterpret_cast<nv_bfloat16*>(w_out.data_ptr()), col, row);
    } else {
      dequantize_iq1_m_r4_matrix_kernel<nv_bfloat16><<<grid, block, 0, stream>>>(
          expert_ptr, reinterpret_cast<nv_bfloat16*>(w_out.data_ptr()), col, row);
    }
  } else {
    TORCH_CHECK(false, "moe_gemm_prefill_tensor only supports fp16/bf16 output");
  }
}

template <typename x_t>
__device__ __forceinline__ half load_x_as_half(const x_t* x, int64_t idx) {
  return static_cast<half>(x[idx]);
}

template <>
__device__ __forceinline__ half load_x_as_half<half>(const half* x, int64_t idx) {
  return x[idx];
}

template <>
__device__ __forceinline__ half load_x_as_half<nv_bfloat16>(const nv_bfloat16* x, int64_t idx) {
  return __float2half(__bfloat162float(x[idx]));
}

template <typename y_t>
__device__ __forceinline__ y_t cast_y(float v);

template <>
__device__ __forceinline__ half cast_y<half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ nv_bfloat16 cast_y<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ float dequant_iq1_s_r4_value(
    const uint8_t* expert_ptr, int n_per_row, int row_idx, int col_idx) {
  const int nblock = n_per_row / 32;
  const int group_bytes = 8 + nblock * static_cast<int>(sizeof(block_iq1_s_r4));
  const int row_group = row_idx / 4;
  const int k = row_idx % 4;
  const int block_idx = col_idx / 32;
  const int j = col_idx % 32;
  const int i = j / 8;
  const int jj = j % 8;

  const uint8_t* group_ptr = expert_ptr + row_group * group_bytes;
  const half* d_scales = reinterpret_cast<const half*>(group_ptr);
  const block_iq1_s_r4* blocks = reinterpret_cast<const block_iq1_s_r4*>(group_ptr + 8);
  const block_iq1_s_r4 blk = blocks[block_idx];
  const uint16_t qh = blk.qh[k];
  const int idx = static_cast<int>(blk.qs[4 * i + k]) | (((qh >> (3 * i)) & 0x07) << 8);
  uint32_t grid32[2];
  grid32[0] = iq1s_grid_gpu[idx];
  grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
  grid32[0] &= 0x0f0f0f0f;
  const int8_t* q = reinterpret_cast<const int8_t*>(grid32);
  const float d = __half2float(d_scales[k]) * (((qh >> 11) & 0x0E) + 1);
  const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
  return d * (float(q[jj]) + delta);
}

__device__ __forceinline__ float dequant_iq1_s_r4_value_lut(
    const uint8_t* expert_ptr, int n_per_row, int row_idx, int col_idx,
    const uint32_t* __restrict__ grid_lut) {
  const int nblock = n_per_row / 32;
  const int group_bytes = 8 + nblock * static_cast<int>(sizeof(block_iq1_s_r4));
  const int row_group = row_idx / 4;
  const int k = row_idx % 4;
  const int block_idx = col_idx / 32;
  const int j = col_idx % 32;
  const int i = j / 8;
  const int jj = j % 8;

  const uint8_t* group_ptr = expert_ptr + row_group * group_bytes;
  const half* d_scales = reinterpret_cast<const half*>(group_ptr);
  const block_iq1_s_r4* blocks = reinterpret_cast<const block_iq1_s_r4*>(group_ptr + 8);
  const block_iq1_s_r4 blk = blocks[block_idx];
  const uint16_t qh = blk.qh[k];
  const int idx = static_cast<int>(blk.qs[4 * i + k]) | (((qh >> (3 * i)) & 0x07) << 8);
  uint32_t grid32[2];
  grid32[0] = grid_lut[idx];
  grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
  grid32[0] &= 0x0f0f0f0f;
  const int8_t* q = reinterpret_cast<const int8_t*>(grid32);
  const float d = __half2float(d_scales[k]) * (((qh >> 11) & 0x0E) + 1);
  const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
  return d * (float(q[jj]) + delta);
}

__device__ __forceinline__ float dequant_iq1_m_r4_value(
    const uint8_t* expert_ptr, int n_per_row, int row_idx, int col_idx) {
  const int nblock = n_per_row / 32;
  const int group_bytes = 8 + nblock * static_cast<int>(sizeof(block_iq1_m_r4));
  const int row_group = row_idx / 4;
  const int k = row_idx % 4;
  const int block_idx = col_idx / 32;
  const int j = col_idx % 32;
  const int i = j / 16;      // 0/1
  const int j16 = j % 16;
  const int h = j16 / 8;     // idx1/idx2
  const int jj = j16 % 8;

  const uint8_t* group_ptr = expert_ptr + row_group * group_bytes;
  const half* d_scales = reinterpret_cast<const half*>(group_ptr);
  const block_iq1_m_r4* blocks = reinterpret_cast<const block_iq1_m_r4*>(group_ptr + 8);
  const block_iq1_m_r4 blk = blocks[block_idx];
  const uint8_t qh_byte = blk.qh[4 * i + k];
  const int idx = (h == 0)
      ? (static_cast<int>(blk.qs[8 * i + k + 0]) | ((qh_byte & 0x07) << 8))
      : (static_cast<int>(blk.qs[8 * i + k + 4]) | ((qh_byte & 0x70) << 4));
  uint32_t grid32[2];
  grid32[0] = iq1s_grid_gpu[idx];
  grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
  grid32[0] &= 0x0f0f0f0f;
  const int8_t* q = reinterpret_cast<const int8_t*>(grid32);
  const float base_d = __half2float(d_scales[k]);
  const float d = (i == 0) ? (base_d * float(blk.scales[k] & 0x0f))
                           : (base_d * float(blk.scales[k] >> 4));
  const float delta = (h == 0)
      ? ((qh_byte & 0x08) ? (-1.0f - IQ1M_DELTA) : (-1.0f + IQ1M_DELTA))
      : ((qh_byte & 0x80) ? (-1.0f - IQ1M_DELTA) : (-1.0f + IQ1M_DELTA));
  return d * (float(q[jj]) + delta);
}

__device__ __forceinline__ float dequant_iq1_m_r4_value_lut(
    const uint8_t* expert_ptr, int n_per_row, int row_idx, int col_idx,
    const uint32_t* __restrict__ grid_lut) {
  const int nblock = n_per_row / 32;
  const int group_bytes = 8 + nblock * static_cast<int>(sizeof(block_iq1_m_r4));
  const int row_group = row_idx / 4;
  const int k = row_idx % 4;
  const int block_idx = col_idx / 32;
  const int j = col_idx % 32;
  const int i = j / 16;      // 0/1
  const int j16 = j % 16;
  const int h = j16 / 8;     // idx1/idx2
  const int jj = j16 % 8;

  const uint8_t* group_ptr = expert_ptr + row_group * group_bytes;
  const half* d_scales = reinterpret_cast<const half*>(group_ptr);
  const block_iq1_m_r4* blocks = reinterpret_cast<const block_iq1_m_r4*>(group_ptr + 8);
  const block_iq1_m_r4 blk = blocks[block_idx];
  const uint8_t qh_byte = blk.qh[4 * i + k];
  const int idx = (h == 0)
      ? (static_cast<int>(blk.qs[8 * i + k + 0]) | ((qh_byte & 0x07) << 8))
      : (static_cast<int>(blk.qs[8 * i + k + 4]) | ((qh_byte & 0x70) << 4));
  uint32_t grid32[2];
  grid32[0] = grid_lut[idx];
  grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
  grid32[0] &= 0x0f0f0f0f;
  const int8_t* q = reinterpret_cast<const int8_t*>(grid32);
  const float base_d = __half2float(d_scales[k]);
  const float d = (i == 0) ? (base_d * float(blk.scales[k] & 0x0f))
                           : (base_d * float(blk.scales[k] >> 4));
  const float delta = (h == 0)
      ? ((qh_byte & 0x08) ? (-1.0f - IQ1M_DELTA) : (-1.0f + IQ1M_DELTA))
      : ((qh_byte & 0x80) ? (-1.0f - IQ1M_DELTA) : (-1.0f + IQ1M_DELTA));
  return d * (float(q[jj]) + delta);
}

template <typename x_t, typename y_t>
__global__ void moe_iq1_r4_wmma_kernel(
    const x_t* __restrict__ X,
    const uint8_t* __restrict__ W,
    y_t* __restrict__ Y,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int* __restrict__ expert_slice_start_dev,
    int64_t type,
    int col,
    int row,
    int max_topk,
    int max_tokens) {
  using namespace nvcuda;
  (void)max_topk;
  const int out_base = blockIdx.y * 16;
  const int col_tile = blockIdx.x * 16;
  if (out_base >= max_tokens || col_tile >= row) return;

  const int idx = __ldg(idx_dev);
  const int limit = __ldg(num_tokens_post_padded_all + idx);
  const int top_k = __ldg(topk_dev);
  const int sorted_start = __ldg(sorted_slice_start_dev);
  const int expert_start = __ldg(expert_slice_start_dev);

  const int row_size = (type == 219) ? (2 + col * 3 / 16) : (2 + col * 7 / 32);
  const int64_t expert_stride = static_cast<int64_t>(row_size) * row;

  __shared__ int s_token_idx[16];
  __shared__ int s_expert_idx[16];
  __shared__ int s_valid[16];
  __shared__ int s_common_expert;
  __shared__ int s_same_expert;
  __shared__ int s_unique_experts[16];
  __shared__ int s_unique_count;
  __shared__ half a_tile[16 * 16];
  __shared__ half b_tile[16 * 16];
  __shared__ float c_tile[16 * 16];

  if (threadIdx.x < 16) {
    const int out_i = out_base + threadIdx.x;
    const int valid = (out_i < limit && out_i < max_tokens) ? 1 : 0;
    s_valid[threadIdx.x] = valid;
    if (valid) {
      const int token_slot = sorted_token_ids_all[sorted_start + out_i];
      s_token_idx[threadIdx.x] = token_slot / top_k;
      s_expert_idx[threadIdx.x] = expert_ids_full[expert_start + out_i];
    } else {
      s_token_idx[threadIdx.x] = 0;
      s_expert_idx[threadIdx.x] = -1;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int common = -1;
    int same = 1;
    int uniq_cnt = 0;
    for (int i = 0; i < 16; ++i) {
      if (!s_valid[i]) continue;
      bool found = false;
      for (int u = 0; u < uniq_cnt; ++u) {
        if (s_unique_experts[u] == s_expert_idx[i]) {
          found = true;
          break;
        }
      }
      if (!found && uniq_cnt < 16) {
        s_unique_experts[uniq_cnt++] = s_expert_idx[i];
      }
      if (common == -1) {
        common = s_expert_idx[i];
      } else if (s_expert_idx[i] != common) {
        same = 0;
      }
    }
    s_unique_count = uniq_cnt;
    s_common_expert = common;
    s_same_expert = same;
  }
  __syncthreads();

  if (s_same_expert && s_common_expert >= 0) {
    const uint8_t* expert_ptr = W + static_cast<int64_t>(s_common_expert) * expert_stride;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < col; k0 += 16) {
      for (int t = threadIdx.x; t < 16 * 16; t += blockDim.x) {
        const int r = t / 16;
        const int c = t % 16;
        a_tile[t] = s_valid[r]
            ? load_x_as_half<x_t>(X, static_cast<int64_t>(s_token_idx[r]) * col + (k0 + c))
            : __float2half(0.0f);
        const int out_col = col_tile + c;
        float wv = 0.0f;
        if (out_col < row) {
          wv = (type == 219)
              ? dequant_iq1_s_r4_value(expert_ptr, col, out_col, k0 + r)
              : dequant_iq1_m_r4_value(expert_ptr, col, out_col, k0 + r);
        }
        b_tile[t] = __float2half(wv);
      }
      __syncthreads();
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
      wmma::load_matrix_sync(a_frag, a_tile, 16);
      wmma::load_matrix_sync(b_frag, b_tile, 16);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      __syncthreads();
    }

    wmma::store_matrix_sync(c_tile, c_frag, 16, wmma::mem_row_major);
    for (int t = threadIdx.x; t < 16 * 16; t += blockDim.x) {
      const int r = t / 16;
      const int c = t % 16;
      const int out_i = out_base + r;
      const int out_col = col_tile + c;
      if (s_valid[r] && out_i < max_tokens && out_col < row) {
        Y[static_cast<int64_t>(out_i) * row + out_col] = cast_y<y_t>(c_tile[t]);
      }
    }
  } else {
    __shared__ float c_acc[16 * 16];
    for (int t = threadIdx.x; t < 16 * 16; t += blockDim.x) {
      c_acc[t] = 0.0f;
    }
    __syncthreads();

    for (int u = 0; u < s_unique_count; ++u) {
      const int expert_idx = s_unique_experts[u];
      if (expert_idx < 0) continue;
      const uint8_t* expert_ptr = W + static_cast<int64_t>(expert_idx) * expert_stride;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag_u;
      wmma::fill_fragment(c_frag_u, 0.0f);
      for (int k0 = 0; k0 < col; k0 += 16) {
        for (int t = threadIdx.x; t < 16 * 16; t += blockDim.x) {
          const int r = t / 16;
          const int c = t % 16;
          a_tile[t] = (s_valid[r] && s_expert_idx[r] == expert_idx)
              ? load_x_as_half<x_t>(X, static_cast<int64_t>(s_token_idx[r]) * col + (k0 + c))
              : __float2half(0.0f);
          const int out_col = col_tile + c;
          float wv = 0.0f;
          if (out_col < row) {
            wv = (type == 219)
                ? dequant_iq1_s_r4_value(expert_ptr, col, out_col, k0 + r)
                : dequant_iq1_m_r4_value(expert_ptr, col, out_col, k0 + r);
          }
          b_tile[t] = __float2half(wv);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag_u;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_u;
        wmma::load_matrix_sync(a_frag_u, a_tile, 16);
        wmma::load_matrix_sync(b_frag_u, b_tile, 16);
        wmma::mma_sync(c_frag_u, a_frag_u, b_frag_u, c_frag_u);
        __syncthreads();
      }
      wmma::store_matrix_sync(c_tile, c_frag_u, 16, wmma::mem_row_major);
      for (int t = threadIdx.x; t < 16 * 16; t += blockDim.x) {
        c_acc[t] += c_tile[t];
      }
      __syncthreads();
    }

    for (int t = threadIdx.x; t < 16 * 16; t += blockDim.x) {
      const int r = t / 16;
      const int c = t % 16;
      const int out_i = out_base + r;
      const int out_col = col_tile + c;
      if (s_valid[r] && out_i < max_tokens && out_col < row) {
        Y[static_cast<int64_t>(out_i) * row + out_col] = cast_y<y_t>(c_acc[t]);
      }
    }
  }
}

__global__ void build_moe_routes_kernel(
    int* __restrict__ route_token_idx,
    int* __restrict__ route_expert_idx,
    int* __restrict__ route_out_idx,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int* __restrict__ expert_slice_start_dev,
    int max_tokens,
    int x_tokens,
    int num_experts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_tokens) return;
  const int idx = __ldg(idx_dev);
  const int limit = __ldg(num_tokens_post_padded_all + idx);
  const int top_k = __ldg(topk_dev);
  const int sorted_start = __ldg(sorted_slice_start_dev);
  const int expert_start = __ldg(expert_slice_start_dev);
  route_out_idx[i] = i;
  if (i < limit) {
    const int token_slot = sorted_token_ids_all[sorted_start + i];
    const int token_idx = token_slot / top_k;
    const int expert_idx = expert_ids_full[expert_start + i];
    if (token_idx >= 0 && token_idx < x_tokens &&
        expert_idx >= 0 && expert_idx < num_experts) {
      route_token_idx[i] = token_idx;
      route_expert_idx[i] = expert_idx;
    } else {
      route_token_idx[i] = 0;
      route_expert_idx[i] = -1;
    }
  } else {
    route_token_idx[i] = 0;
    route_expert_idx[i] = -1;
  }
}

__global__ void count_expert_routes_kernel(
    const int* __restrict__ route_expert_idx,
    int* __restrict__ expert_counts,
    int max_tokens,
    int num_experts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_tokens) return;
  const int e = route_expert_idx[i];
  if (e >= 0 && e < num_experts) {
    atomicAdd(expert_counts + e, 1);
  }
}

__global__ void scatter_expert_routes_kernel(
    const int* __restrict__ route_token_idx,
    const int* __restrict__ route_expert_idx,
    const int* __restrict__ route_out_idx,
    const int* __restrict__ expert_offsets,
    int* __restrict__ expert_cursors,
    int* __restrict__ grouped_token_idx,
    int* __restrict__ grouped_out_idx,
    int max_tokens,
    int num_experts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_tokens) return;
  const int e = route_expert_idx[i];
  if (e < 0 || e >= num_experts) return;
  const int local = atomicAdd(expert_cursors + e, 1);
  const int dst = expert_offsets[e] + local;
  grouped_token_idx[dst] = route_token_idx[i];
  grouped_out_idx[dst] = route_out_idx[i];
}

__global__ void build_expert_tile_map_kernel(
    const int* __restrict__ expert_tile_offsets,
    const int* __restrict__ expert_tile_counts,
    int* __restrict__ block_expert_idx,
    int* __restrict__ block_token_tile_idx,
    int num_experts) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= num_experts) return;
  const int off = expert_tile_offsets[e];
  const int cnt = expert_tile_counts[e];
  for (int t = 0; t < cnt; ++t) {
    block_expert_idx[off + t] = e;
    block_token_tile_idx[off + t] = t * 16;
  }
}

__global__ void compute_expert_tile_counts_kernel(
    const int* __restrict__ expert_counts,
    int* __restrict__ expert_tile_counts,
    int num_experts) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= num_experts) return;
  const int c = expert_counts[e];
  expert_tile_counts[e] = (c + 15) / 16;
}

template <typename x_t, typename y_t>
__global__ void moe_iq1_r4_grouped_wmma_kernel(
    const x_t* __restrict__ X,
    const uint8_t* __restrict__ W,
    y_t* __restrict__ Y,
    const int* __restrict__ grouped_token_idx,
    const int* __restrict__ grouped_out_idx,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ expert_counts,
    const int* __restrict__ block_expert_idx,
    const int* __restrict__ block_token_tile_idx,
    const int* __restrict__ total_tiles_dev,
    int64_t type,
    int col,
    int row,
    int x_tokens,
    int num_experts,
    int max_tokens) {
  using namespace nvcuda;
  constexpr int TILES_PER_BLOCK = 1;
  const int tile_linear = blockIdx.y;
  const int col_tile = blockIdx.x * 16;
  if (col_tile >= row) return;
  const int total_tiles = __ldg(total_tiles_dev);
  const int tile_begin = tile_linear * TILES_PER_BLOCK;
  if (tile_begin >= total_tiles) return;
  const int expert = block_expert_idx[tile_begin];
  if (expert < 0 || expert >= num_experts) return;

  const int count = expert_counts[expert];
  const int expert_base = expert_offsets[expert];

  const int row_size = (type == 219) ? (2 + col * 3 / 16) : (2 + col * 7 / 32);
  const int64_t expert_stride = static_cast<int64_t>(row_size) * row;
  const uint8_t* expert_ptr = W + static_cast<int64_t>(expert) * expert_stride;

  __shared__ int s_token_idx[TILES_PER_BLOCK][16];
  __shared__ int s_out_idx[TILES_PER_BLOCK][16];
  __shared__ int s_valid[TILES_PER_BLOCK][16];
  __shared__ uint32_t s_iq1_grid[2048];
  __shared__ half a_tile[TILES_PER_BLOCK][16 * 16];
  __shared__ half b_tile[16 * 16];
  __shared__ float c_tile[TILES_PER_BLOCK][16 * 16];

  for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
    s_iq1_grid[i] = iq1s_grid_gpu[i];
  }
  __syncthreads();

  if (threadIdx.x < 16 * TILES_PER_BLOCK) {
    const int tile_id = threadIdx.x / 16;
    const int r = threadIdx.x % 16;
    const int map_idx = tile_begin + tile_id;
    int valid = 0;
    int token_idx = 0;
    int out_idx = 0;
    if (map_idx < total_tiles && block_expert_idx[map_idx] == expert) {
      const int token_tile = block_token_tile_idx[map_idx];
      valid = (token_tile + r < count) ? 1 : 0;
      if (valid) {
        const int base = expert_base + token_tile;
        token_idx = grouped_token_idx[base + r];
        out_idx = grouped_out_idx[base + r];
        if (token_idx < 0 || token_idx >= x_tokens) {
          valid = 0;
          token_idx = 0;
        }
      }
    }
    s_valid[tile_id][r] = valid;
    s_token_idx[tile_id][r] = token_idx;
    s_out_idx[tile_id][r] = out_idx;
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[TILES_PER_BLOCK];
  #pragma unroll
  for (int tb = 0; tb < TILES_PER_BLOCK; ++tb) {
    wmma::fill_fragment(c_frag[tb], 0.0f);
  }
  for (int k0 = 0; k0 < col; k0 += 16) {
    for (int t = threadIdx.x; t < 16 * 16; t += blockDim.x) {
      const int r = t / 16;
      const int c = t % 16;
      const int out_col = col_tile + c;
      float wv = 0.0f;
      if (out_col < row) {
        wv = (type == 219)
            ? dequant_iq1_s_r4_value_lut(expert_ptr, col, out_col, k0 + r, s_iq1_grid)
            : dequant_iq1_m_r4_value_lut(expert_ptr, col, out_col, k0 + r, s_iq1_grid);
      }
      b_tile[t] = __float2half(wv);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::load_matrix_sync(b_frag, b_tile, 16);

    #pragma unroll
    for (int tb = 0; tb < TILES_PER_BLOCK; ++tb) {
      for (int t = threadIdx.x; t < 16 * 16; t += blockDim.x) {
        const int r = t / 16;
        const int c = t % 16;
        a_tile[tb][t] = s_valid[tb][r]
            ? load_x_as_half<x_t>(X, static_cast<int64_t>(s_token_idx[tb][r]) * col + (k0 + c))
            : __float2half(0.0f);
      }
      __syncthreads();
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::load_matrix_sync(a_frag, a_tile[tb], 16);
      wmma::mma_sync(c_frag[tb], a_frag, b_frag, c_frag[tb]);
      __syncthreads();
    }
    __syncthreads();
  }

  #pragma unroll
  for (int tb = 0; tb < TILES_PER_BLOCK; ++tb) {
    wmma::store_matrix_sync(c_tile[tb], c_frag[tb], 16, wmma::mem_row_major);
    for (int t = threadIdx.x; t < 16 * 16; t += blockDim.x) {
      const int r = t / 16;
      const int c = t % 16;
      const int out_col = col_tile + c;
      if (s_valid[tb][r] && out_col < row) {
        const int out_i = s_out_idx[tb][r];
        if (out_i < max_tokens) {
          Y[static_cast<int64_t>(out_i) * row + out_col] = cast_y<y_t>(c_tile[tb][t]);
        }
      }
    }
  }
}

torch::Tensor moe_gemm_prefill_tensor(
  torch::Tensor X,
  torch::Tensor W,
  torch::Tensor sorted_token_ids_all,
  torch::Tensor expert_ids,
  torch::Tensor num_tokens_post_padded_all,
  int64_t type,
  int64_t row,
  torch::Tensor idx_dev,
  torch::Tensor topk_dev,
  torch::Tensor sorted_slice_start_dev,
  torch::Tensor expert_slice_start_dev,
  int64_t max_topk,
  int64_t max_tokens
) {
  (void)num_tokens_post_padded_all;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
  TORCH_CHECK(sorted_token_ids_all.is_cuda(), "sorted_token_ids_all must be CUDA");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be CUDA");
  TORCH_CHECK(type == 219 || type == 229, "moe_gemm_prefill_tensor only supports IQ1_*_R4 (219/229)");
  TORCH_CHECK(row % 4 == 0, "row must be divisible by 4");
  TORCH_CHECK(X.scalar_type() == torch::kFloat16 || X.scalar_type() == torch::kBFloat16,
              "moe_gemm_prefill_tensor expects fp16/bf16 activations");

  const int64_t x_tokens = X.size(0);
  const int64_t col = X.size(1);
  auto options = torch::TensorOptions().dtype(X.dtype()).device(X.device());
  at::Tensor Y = torch::zeros({max_tokens, row}, options);
  if (max_tokens == 0 || x_tokens == 0) {
    return Y;
  }

  TORCH_CHECK(topk_dev.is_cuda() && sorted_slice_start_dev.is_cuda() &&
              expert_slice_start_dev.is_cuda() && idx_dev.is_cuda() &&
              num_tokens_post_padded_all.is_cuda(),
              "routing metadata tensors must be CUDA");
  TORCH_CHECK(sorted_token_ids_all.is_cuda() && expert_ids.is_cuda(),
              "sorted_token_ids_all/expert_ids must be CUDA");
  TORCH_CHECK(col % 16 == 0, "col must be multiple of 16 for wmma kernel");

  const int row_size = (type == 219) ? (2 + static_cast<int>(col) * 3 / 16)
                                     : (2 + static_cast<int>(col) * 7 / 32);
  const int64_t expert_stride = static_cast<int64_t>(row_size) * row;
  TORCH_CHECK(W.numel() % expert_stride == 0, "W size is not divisible by expert_stride");
  const int num_experts = static_cast<int>(W.numel() / expert_stride);
  TORCH_CHECK(num_experts > 0, "num_experts must be > 0");

  auto i32opt = torch::TensorOptions().dtype(torch::kInt32).device(X.device());
  at::Tensor route_token_idx = torch::empty({max_tokens}, i32opt);
  at::Tensor route_expert_idx = torch::empty({max_tokens}, i32opt);
  at::Tensor route_out_idx = torch::empty({max_tokens}, i32opt);
  at::Tensor expert_counts = torch::zeros({num_experts}, i32opt);
  at::Tensor expert_cursors = torch::zeros({num_experts}, i32opt);
  at::Tensor grouped_token_idx = torch::empty({max_tokens}, i32opt);
  at::Tensor grouped_out_idx = torch::empty({max_tokens}, i32opt);

  dim3 block(256, 1, 1);
  dim3 route_grid((max_tokens + 255) / 256, 1, 1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  build_moe_routes_kernel<<<route_grid, block, 0, stream>>>(
      route_token_idx.data_ptr<int>(),
      route_expert_idx.data_ptr<int>(),
      route_out_idx.data_ptr<int>(),
      sorted_token_ids_all.data_ptr<int>(),
      expert_ids.data_ptr<int>(),
      num_tokens_post_padded_all.data_ptr<int>(),
      idx_dev.data_ptr<int>(),
      topk_dev.data_ptr<int>(),
      sorted_slice_start_dev.data_ptr<int>(),
      expert_slice_start_dev.data_ptr<int>(),
      static_cast<int>(max_tokens),
      static_cast<int>(x_tokens),
      num_experts);
  count_expert_routes_kernel<<<route_grid, block, 0, stream>>>(
      route_expert_idx.data_ptr<int>(),
      expert_counts.data_ptr<int>(),
      static_cast<int>(max_tokens),
      num_experts);

  at::Tensor expert_offsets = torch::zeros({num_experts + 1}, i32opt);
  at::Tensor counts_cumsum = expert_counts.cumsum(0, torch::kInt32);
  expert_offsets.narrow(0, 1, num_experts).copy_(counts_cumsum);

  scatter_expert_routes_kernel<<<route_grid, block, 0, stream>>>(
      route_token_idx.data_ptr<int>(),
      route_expert_idx.data_ptr<int>(),
      route_out_idx.data_ptr<int>(),
      expert_offsets.data_ptr<int>(),
      expert_cursors.data_ptr<int>(),
      grouped_token_idx.data_ptr<int>(),
      grouped_out_idx.data_ptr<int>(),
      static_cast<int>(max_tokens),
      num_experts);

  at::Tensor expert_tile_counts = torch::zeros({num_experts}, i32opt);
  dim3 tile_count_grid((num_experts + 255) / 256, 1, 1);
  compute_expert_tile_counts_kernel<<<tile_count_grid, block, 0, stream>>>(
      expert_counts.data_ptr<int>(),
      expert_tile_counts.data_ptr<int>(),
      num_experts);
  at::Tensor expert_tile_offsets = torch::zeros({num_experts + 1}, i32opt);
  at::Tensor tile_cumsum = expert_tile_counts.cumsum(0, torch::kInt32);
  expert_tile_offsets.narrow(0, 1, num_experts).copy_(tile_cumsum);
  at::Tensor total_tiles_dev = tile_cumsum.narrow(0, num_experts - 1, 1).clone();
  const int max_tile_slots = static_cast<int>(max_tokens) + num_experts;
  at::Tensor block_expert_idx = torch::full({max_tile_slots}, -1, i32opt);
  at::Tensor block_token_tile_idx = torch::zeros({max_tile_slots}, i32opt);
  dim3 tile_map_grid((num_experts + 255) / 256, 1, 1);
  build_expert_tile_map_kernel<<<tile_map_grid, block, 0, stream>>>(
      expert_tile_offsets.data_ptr<int>(),
      expert_tile_counts.data_ptr<int>(),
      block_expert_idx.data_ptr<int>(),
      block_token_tile_idx.data_ptr<int>(),
      num_experts);

  dim3 wmma_block(32, 1, 1);
  constexpr int TILES_PER_BLOCK = 1;
  dim3 wmma_grid((row + 15) / 16, (max_tile_slots + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK, 1);
  if (X.scalar_type() == torch::kFloat16) {
    moe_iq1_r4_grouped_wmma_kernel<half, half><<<wmma_grid, wmma_block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr()),
        reinterpret_cast<const uint8_t*>(W.data_ptr()),
        reinterpret_cast<half*>(Y.data_ptr()),
        grouped_token_idx.data_ptr<int>(),
        grouped_out_idx.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        expert_counts.data_ptr<int>(),
        block_expert_idx.data_ptr<int>(),
        block_token_tile_idx.data_ptr<int>(),
        total_tiles_dev.data_ptr<int>(),
        type, static_cast<int>(col), static_cast<int>(row),
        static_cast<int>(x_tokens), num_experts, static_cast<int>(max_tokens));
  } else {
    moe_iq1_r4_grouped_wmma_kernel<nv_bfloat16, nv_bfloat16><<<wmma_grid, wmma_block, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(X.data_ptr()),
        reinterpret_cast<const uint8_t*>(W.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(Y.data_ptr()),
        grouped_token_idx.data_ptr<int>(),
        grouped_out_idx.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        expert_counts.data_ptr<int>(),
        block_expert_idx.data_ptr<int>(),
        block_token_tile_idx.data_ptr<int>(),
        total_tiles_dev.data_ptr<int>(),
        type, static_cast<int>(col), static_cast<int>(row),
        static_cast<int>(x_tokens), num_experts, static_cast<int>(max_tokens));
  }
  return Y;
}

torch::Tensor moe_gemm_prefill( 
  torch::Tensor X,                      // input:   [token_num,hidden_size]
  torch::Tensor W,                      // Weights: quantized and packed weighst
  torch::Tensor sorted_token_ids_all,       // [token_num], [0,topk,topk*2,...,topk*(tok_num-1)]
  torch::Tensor expert_ids,             // [token_num * max_topk]，
  torch::Tensor num_tokens_post_padded_all, // [1] = token_num * topk
  int64_t type, int64_t row,
  torch::Tensor idx_dev, 
  torch::Tensor topk_dev, 
  torch::Tensor sorted_slice_start_dev,
  torch::Tensor expert_slice_start_dev,
  int64_t max_topk,
  int64_t max_tokens
) {
  using torch::indexing::Slice;
  int x_tokens = X.sizes()[0];
  constexpr int kBlockSize = 128;
  int col = X.sizes()[1];
  int padded = (col + kBlockSize - 1) / kBlockSize * kBlockSize;
  assert(row % 4 == 0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({max_tokens, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({x_tokens, padded / kBlockSize * 35}, options);
  const auto token_num = x_tokens;
  // p1: expert_slice_start=0, top_k=flex_topk; p2: expert_slice_start=flex_topk, top_k=(max_topk-flex_topk)
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe", [&] {
    quantize_row_q8_k128_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                          col, x_tokens, stream);
    switch (type) {
      case 219:{
        const int row_size = 2 + col * 1.5 / 8;
        const int expert_stride = row_size * row;
        ggml_moe_vec_iq1_s_r4_q8_k128_cuda_dyn_prefill(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(), (scalar_t*)Y.data_ptr(),
            sorted_token_ids_all.data_ptr<int>(),
            expert_ids.data_ptr<int>(),
            num_tokens_post_padded_all.data_ptr<int>(),
            idx_dev.data_ptr<int>(),
            topk_dev.data_ptr<int>(),
            sorted_slice_start_dev.data_ptr<int>(),
            expert_slice_start_dev.data_ptr<int>(),
            expert_stride, col, row,
            (int)max_tokens, padded, row,
            (int)max_topk, (int)max_tokens, stream);
        break;
      }
      case 229:{
        const int row_size = 2 + col *  7 / 32;
        const int expert_stride = row_size * row;
        ggml_moe_vec_iq1_m_r4_q8_k128_cuda_dyn_prefill(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(), (scalar_t*)Y.data_ptr(),
            sorted_token_ids_all.data_ptr<int>(),
            expert_ids.data_ptr<int>(),
            num_tokens_post_padded_all.data_ptr<int>(),
            idx_dev.data_ptr<int>(),
            topk_dev.data_ptr<int>(),
            sorted_slice_start_dev.data_ptr<int>(),
            expert_slice_start_dev.data_ptr<int>(),
            expert_stride, col, row,
            (int)max_tokens, padded, row,
            (int)max_topk, (int)max_tokens, stream);
        break;
      }
    }
  });
  return Y;
}

torch::Tensor moe_gemm_w2_prefill( 
  torch::Tensor X,                      // input:   [token_num,hidden_size]
  torch::Tensor W,                      // Weights: quantized and packed weighst
  torch::Tensor sorted_token_ids_all,       // [token_num], [0,topk,topk*2,...,topk*(tok_num-1)]
  torch::Tensor expert_ids,             // [token_num * max_topk]，
  torch::Tensor num_tokens_post_padded_all, // [1] = token_num * topk
  int64_t type, int64_t row, 
  int64_t top_k,
  torch::Tensor idx_dev,
  torch::Tensor expert_slice_start_dev,
  torch::Tensor sorted_slice_start_dev,
  int64_t max_topk,
  int64_t max_tokens
  ) {
    constexpr int kBlockSize = 128;
    int col = X.sizes()[1];
    int x_tokens = X.sizes()[0];
    int padded = (col + kBlockSize - 1) / kBlockSize * kBlockSize;
    assert(row % 4 == 0);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
    auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
    at::Tensor Y = torch::empty({max_tokens, row}, options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
    at::Tensor quant_X = torch::empty({x_tokens, padded / kBlockSize * 35}, options);
    VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe", [&] {
      quantize_row_q8_k128_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                            col, x_tokens, stream);
    switch (type) {
      case 219:{
        const int row_size = 2 + col * 1.5 / 8;
        const int expert_stride = row_size * row;
        ggml_moe_vec_iq1_s_r4_q8_k128_cuda_dyn_w2_prefill(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(), (scalar_t*)Y.data_ptr(),
            sorted_token_ids_all.data_ptr<int>(),
            expert_ids.data_ptr<int>(),
            num_tokens_post_padded_all.data_ptr<int>(),
            idx_dev.data_ptr<int>(),
            expert_slice_start_dev.data_ptr<int>(),
            sorted_slice_start_dev.data_ptr<int>(),
            expert_stride, col, row,
            (int)max_tokens, padded, row,
            (int)top_k,
            (int)max_topk, (int)max_tokens, stream);
        break;
      }
      case 229:{
        const int row_size = 2 + col *  7 / 32;
        const int expert_stride = row_size * row;
        ggml_moe_vec_iq1_m_r4_q8_k128_cuda_dyn_w2_prefill(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(), (scalar_t*)Y.data_ptr(),
            sorted_token_ids_all.data_ptr<int>(),
            expert_ids.data_ptr<int>(),
            num_tokens_post_padded_all.data_ptr<int>(),
            idx_dev.data_ptr<int>(),
            expert_slice_start_dev.data_ptr<int>(),
            sorted_slice_start_dev.data_ptr<int>(),
            expert_stride, col, row,
            (int)max_tokens, padded, row,
            (int)top_k,
            (int)max_topk, (int)max_tokens, stream);
        break;
      }
    }
  });
  return Y;
  }