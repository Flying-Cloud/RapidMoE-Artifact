// adapted from https://github.com/jinzhen-lin/vllm/blob/ea3970282137f8dcc04d3265680a3366408494fe/csrc/moe/moe_align_sum_kernels.cu
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace vllm {
namespace moe {

namespace {
__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row,
                                         int32_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}
}  // namespace

template <typename scalar_t, typename token_cnts_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids,
                                            int32_t* sorted_token_ids,
                                            int32_t* expert_ids,
                                            int32_t* total_tokens_post_pad,
                                            int32_t num_experts,
                                            int32_t block_size, size_t numel) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;  // 1d tensor with shape (num_experts + 1)
  token_cnts_t* tokens_cnts =
      (token_cnts_t*)(shared_mem + num_experts +
                      1);  // 2d tensor with shape (blockDim.x + 1, num_experts)

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  if (threadIdx.x < num_experts) {
    tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[index(num_experts, i, threadIdx.x)] +=
          tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
                  CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)],
                          block_size) *
                      block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  /**
   * Each thread processes a token shard, calculating the index of each token
   * after sorting by expert number. Given the example topk_ids =
   * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
   * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
   * padding value(preset in python).
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int32_t rank_post_pad =
        tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
        cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
  }
}

// TODO(simon): this is temporarily adapted from
// https://github.com/sgl-project/sglang/commit/31548116a8dc8c6df7e146e0587335a59fc5b9d7
// we did this to unblock Deepseek V3 but there should be a better
// implementation to manage shared memory.
template <typename scalar_t>
__global__ void moe_align_block_size_global_mem_kernel(
    scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
    int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, int32_t* tokens_cnts, int32_t* cumsum) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  if (threadIdx.x < num_experts) {
    tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[index(num_experts, i, threadIdx.x)] +=
          tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
                  CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)],
                          block_size) *
                      block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  /**
   * Each thread processes a token shard, calculating the index of each token
   * after sorting by expert number. Given the example topk_ids =
   * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
   * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
   * padding value(preset in python).
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int32_t rank_post_pad =
        tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
        cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
  }
}

// taken from
// https://github.com/sgl-project/sglang/commit/cdae77b03dfc6fec3863630550b45bbfc789f957
template <typename scalar_t>
__global__ void sgl_moe_align_block_size_kernel(
    scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
    int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, int32_t* cumsum) {
  __shared__ int32_t shared_counts[32][8];

  const int warp_id = threadIdx.x / 32;
  const int experts_per_warp = 8;
  const int my_expert_start = warp_id * experts_per_warp;

  // Initialize shared_counts for this warp's experts
  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < num_experts) {
      shared_counts[warp_id][i] = 0;
    }
  }

  __syncthreads();

  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
  }

  __syncthreads();

  // Single thread computes cumulative sum and total tokens
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx][expert_offset];

      cumsum[i] =
          cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  // Assign expert IDs to blocks
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
}

// taken from
// https://github.com/sgl-project/sglang/commit/cdae77b03dfc6fec3863630550b45bbfc789f957
template <typename scalar_t>
__global__ void sgl_moe_token_sort_kernel(scalar_t* __restrict__ topk_ids,
                                          int32_t* sorted_token_ids,
                                          int32_t* cumsum_buffer,
                                          size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_weight_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    float * __restrict__ weight,       // [topk]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]) * weight[k];
    }
    out[token_idx * d + idx] = (scalar_t)x;
  }
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_weight_kernel_d1(
    scalar_t* __restrict__ out,          // [..., d]
    float * __restrict__ weight,       // [..., topk]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d,
    const int d1) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]) * weight[k + token_idx * d1];
    }
    out[token_idx * d + idx] = (scalar_t)x;
  }
}

template <typename scalar_t>
__global__ void moe_sum_weight_kernel_d1_dyn(
    scalar_t* __restrict__ out,                 // [num_tokens, d]
    const float* __restrict__ weight_full,      // [num_tokens, max_topk] (or [max_topk] when num_tokens==1)
    const scalar_t* __restrict__ input,         // [num_tokens, topk, d] where topk = max_topk - flex_topk
    const int d,
    const int max_topk,
    const int* __restrict__ flex_topk_dev,
    const int num_tokens) {
  const int flex_topk = __ldg(flex_topk_dev);
  const int topk = max_topk - flex_topk;
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const float* w = (num_tokens > 1)
                       ? (weight_full + token_idx * max_topk + flex_topk)
                       : (weight_full + flex_topk);

  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = 0.0f;
    const int64_t base = token_idx * (int64_t)topk * d + idx;
    for (int k = 0; k < topk; ++k) {
      x += (float)VLLM_LDG(&input[base + (int64_t)k * d]) * w[k];
    }
    out[token_idx * d + idx] = (scalar_t)x;
  }
}

template <typename scalar_t>
__global__ void dynamic_add_kernel(
    scalar_t* __restrict__ input1,      // [num_tokens, hidden_size]
    const scalar_t* __restrict__ input2,  // [num_tokens, hidden_size]
    const int* __restrict__ flex_topk_dev, // scalar tensor on device
    const int num_tokens,
    const int hidden_size) {
  // 若 k == 0，则无需进行加和
  const int flex_topk = __ldg(flex_topk_dev);
  const int token_idx = (int)blockIdx.x;
  if (token_idx >= num_tokens) return;
  const int64_t base = (int64_t)token_idx * hidden_size;
  if (flex_topk == 0) {
    for (int idx = (int)threadIdx.x; idx < hidden_size; idx += (int)blockDim.x) {
      input1[base + idx] = input2[base + idx];
    }
  }
  else {
    for (int idx = (int)threadIdx.x; idx < hidden_size; idx += (int)blockDim.x) {
      input1[base + idx] += input2[base + idx];
    }
  }
}

__global__ void dynamic_threshold_reduce_kernel(
    const float* __restrict__ topk_weight,
    float* __restrict__ avg_weight,
    const int* __restrict__ bsz_tensor,
    const int num_experts) {
  const int expert_idx = (int)threadIdx.x;
  const int actual_tokens = __ldg(bsz_tensor);
  if (expert_idx >= num_experts) return;
  float acc = 0.0f;
  for (int token_idx = 0; token_idx < actual_tokens; ++token_idx) {
    acc += VLLM_LDG(&topk_weight[token_idx * num_experts + expert_idx]);
  }
  avg_weight[expert_idx] = acc / (float)max(1, actual_tokens);
}

__global__ void dynamic_threshold_decide_kernel(
    const float* __restrict__ avg_weight,
    const float* __restrict__ alpha,
    const float* __restrict__ threshold,
    int* __restrict__ flex_topk_dev,
    int* __restrict__ flex_idx_dev) {
  const float a = __ldg(alpha);
  const float tau = __ldg(threshold);
  const float s1 = a * VLLM_LDG(&avg_weight[0]);
  const float s2 = a * VLLM_LDG(&avg_weight[1]);
  const float s3 = a * VLLM_LDG(&avg_weight[2]);
  const float s4 = a * VLLM_LDG(&avg_weight[3]);
  const float s5 = a * VLLM_LDG(&avg_weight[4]);
  const float s6 = a * VLLM_LDG(&avg_weight[5]);
  int selected_topk = 1;
  if (s1 < tau) {
    selected_topk = 0;
  } else if (s5 > tau) {
    selected_topk = 6;
  } else if (s1 > tau && s2 < tau) {
    selected_topk = 1;
  } else if (s2 > tau && s3 < tau) {
    selected_topk = 2;
  } else if (s3 > tau && s4 < tau) {
    selected_topk = 3;
  } else if (s4 > tau && s5 < tau) {
    selected_topk = 4;
  } else {
    selected_topk = 1;
  }
  flex_topk_dev[0] = selected_topk;
  flex_idx_dev[0] = selected_topk;
  //printf("selected_topk: %d\n", selected_topk);
  //printf("s1: %f, s2: %f, s3: %f, s4: %f\n", s1, s2, s3, s4);
  //printf("tau: %f, a: %f\n", tau, a);
  //printf("avg_weight[0]: %f, avg_weight[1]: %f, avg_weight[2]: %f, avg_weight[3]: %f\n", VLLM_LDG(&avg_weight[0]), VLLM_LDG(&avg_weight[1]), VLLM_LDG(&avg_weight[2]), VLLM_LDG(&avg_weight[3]));
  //printf("flex_topk_dev[0]: %d, flex_idx_dev[0]: %d\n", flex_topk_dev[0], flex_idx_dev[0]);
}

}  // namespace moe
}  // namespace vllm
void moe_align_block_size_v1(torch::Tensor topk_ids_raw, int64_t topk_start, int64_t topk_end, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int device_max_shared_mem;
  // 根据flex_topk，对topk_ids进行整理
  using torch::indexing::Slice;
  int token_num = topk_ids_raw.sizes()[0];
  auto topk_ids = topk_ids_raw.index({Slice(),Slice(topk_start, topk_end)}).contiguous().view({token_num, -1});
  auto dev = topk_ids.get_device();
  cudaDeviceGetAttribute(&device_max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

  const int32_t num_thread = max((int32_t)num_experts, WARP_SIZE);
  const int32_t shared_mem_i32 =
      ((num_thread + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
  const int32_t shared_mem_i16 =
      ((num_thread + 1) * num_experts) * sizeof(uint16_t) +
      (num_experts + 1) * sizeof(int32_t);

  bool use_global_memory = false;
  bool use_i16 = false;  // Use uint16_t for shared memory token counts
  if (shared_mem_i32 < device_max_shared_mem) {
    // Do nothing in this case. We're all set to use int32_t token counts
  } else if (shared_mem_i16 < device_max_shared_mem &&
             topk_ids.numel() <= 65535) {
    // when nelements of topk_ids is smaller than 65535 (max value of uint16),
    // element value of token_cnts would also smaller than 65535,
    // so we can use uint16 as dtype of token_cnts
    use_i16 = true;
  } else {
    use_global_memory = true;
  }

  if (use_global_memory) {
    VLLM_DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_global_mem_kernel", [&] {
          // calc needed amount of shared mem for `tokens_cnts` and `cumsum`
          // tensors
          const int32_t num_thread = max((int32_t)num_experts, WARP_SIZE);

          auto options_int = torch::TensorOptions()
                                 .dtype(torch::kInt)
                                 .device(topk_ids.device());
          torch::Tensor token_cnts_buffer =
              torch::empty({(num_experts + 1) * num_experts}, options_int);
          torch::Tensor cumsum_buffer =
              torch::empty({num_experts + 1}, options_int);

          auto kernel =
              vllm::moe::moe_align_block_size_global_mem_kernel<scalar_t>;
          kernel<<<1, num_thread, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel(), token_cnts_buffer.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>());
        });
  } else if (use_i16) {
    VLLM_DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
          // set dynamic shared mem
          auto kernel =
              vllm::moe::moe_align_block_size_kernel<scalar_t, uint16_t>;
          AT_CUDA_CHECK(VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
              (void*)kernel, shared_mem_i16));
          kernel<<<1, num_thread, shared_mem_i16, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel());
        });
  } else {
    VLLM_DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
          auto kernel =
              vllm::moe::moe_align_block_size_kernel<scalar_t, int32_t>;
          AT_CUDA_CHECK(VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
              (void*)kernel, shared_mem_i32));
          kernel<<<1, num_thread, shared_mem_i32, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel());
        });
  }
}

template <typename scalar_t>
__global__ void dynamic_add_kernel(
    scalar_t* __restrict__ input1,      // [num_tokens, hidden_size]
    const scalar_t* __restrict__ input2, // [num_tokens, hidden_size]
    const int* __restrict__ flex_topk_dev,
    int num_tokens,
    int hidden_size) {
  
  const int flex_topk = __ldg(flex_topk_dev);
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;
  // 如果 flex_topk == 0，直接返回，不执行任何操作
  if (flex_topk == 0) {
    for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
      input1[token_idx * hidden_size + idx] = input2[token_idx * hidden_size + idx];
    }
  }
  else {
    for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
      input1[token_idx * hidden_size + idx] += input2[token_idx * hidden_size + idx];
    }
  }

}

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int device_max_shared_mem;
  auto dev = topk_ids.get_device();
  cudaDeviceGetAttribute(&device_max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

  const int32_t num_thread = max((int32_t)num_experts, WARP_SIZE);
  const int32_t shared_mem_i32 =
      ((num_thread + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
  const int32_t shared_mem_i16 =
      ((num_thread + 1) * num_experts) * sizeof(uint16_t) +
      (num_experts + 1) * sizeof(int32_t);

  bool use_global_memory = false;
  bool use_i16 = false;  // Use uint16_t for shared memory token counts
  if (shared_mem_i32 < device_max_shared_mem) {
    // Do nothing in this case. We're all set to use int32_t token counts
  } else if (shared_mem_i16 < device_max_shared_mem &&
             topk_ids.numel() <= 65535) {
    // when nelements of topk_ids is smaller than 65535 (max value of uint16),
    // element value of token_cnts would also smaller than 65535,
    // so we can use uint16 as dtype of token_cnts
    use_i16 = true;
  } else {
    use_global_memory = true;
  }

  if (use_global_memory) {
    VLLM_DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_global_mem_kernel", [&] {
          // calc needed amount of shared mem for `tokens_cnts` and `cumsum`
          // tensors
          const int32_t num_thread = max((int32_t)num_experts, WARP_SIZE);

          auto options_int = torch::TensorOptions()
                                 .dtype(torch::kInt)
                                 .device(topk_ids.device());
          torch::Tensor token_cnts_buffer =
              torch::empty({(num_experts + 1) * num_experts}, options_int);
          torch::Tensor cumsum_buffer =
              torch::empty({num_experts + 1}, options_int);

          auto kernel =
              vllm::moe::moe_align_block_size_global_mem_kernel<scalar_t>;
          kernel<<<1, num_thread, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel(), token_cnts_buffer.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>());
        });
  } else if (use_i16) {
    VLLM_DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
          // set dynamic shared mem
          auto kernel =
              vllm::moe::moe_align_block_size_kernel<scalar_t, uint16_t>;
          AT_CUDA_CHECK(VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
              (void*)kernel, shared_mem_i16));
          kernel<<<1, num_thread, shared_mem_i16, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel());
        });
  } else {
    VLLM_DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
          auto kernel =
              vllm::moe::moe_align_block_size_kernel<scalar_t, int32_t>;
          AT_CUDA_CHECK(VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
              (void*)kernel, shared_mem_i32));
          kernel<<<1, num_thread, shared_mem_i32, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel());
        });
  }
}

void sgl_moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                              int64_t block_size,
                              torch::Tensor sorted_token_ids,
                              torch::Tensor experts_ids,
                              torch::Tensor num_tokens_post_pad) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(num_experts == 256,
              "sgl_moe_align_block_size kernel only supports deepseek v3.");

  VLLM_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "sgl_moe_align_block_size_kernel", [&] {
        // calc needed amount of shared mem for `cumsum` tensors
        auto options_int =
            torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
        torch::Tensor cumsum_buffer =
            torch::zeros({num_experts + 1}, options_int);

        auto align_kernel =
            vllm::moe::sgl_moe_align_block_size_kernel<scalar_t>;
        align_kernel<<<1, 1024, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
            experts_ids.data_ptr<int32_t>(),
            num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
            topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>());

        const int block_threads = 256;
        const int num_blocks =
            (topk_ids.numel() + block_threads - 1) / block_threads;
        const int max_blocks = 65535;
        const int actual_blocks = std::min(num_blocks, max_blocks);
        auto sort_kernel = vllm::moe::sgl_moe_token_sort_kernel<scalar_t>;
        sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel());
      });
}

void moe_sum(torch::Tensor& input,   // [num_tokens, topk, hidden_size]
             torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const int num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (topk) {
    case 2:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    case 3:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    case 4:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}

void moe_weight_sum(torch::Tensor& input,   // [num_tokens, topk, hidden_size]
  torch::Tensor& weight,  // [nt, topk]
  torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const int num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);
  const int d1 = (input.size(0) > 1) ? topk : 0;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (topk) {
  case 2:
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_weight_kernel_d1", [&] {
  vllm::moe::moe_sum_weight_kernel_d1<scalar_t, 2><<<grid, block, 0, stream>>>(
  output.data_ptr<scalar_t>(), weight.data_ptr<float>(),input.data_ptr<scalar_t>(),
  hidden_size,d1);
  });
  break;

  case 3:
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_weight_kernel_d1", [&] {
  vllm::moe::moe_sum_weight_kernel_d1<scalar_t, 3><<<grid, block, 0, stream>>>(
  output.data_ptr<scalar_t>(), weight.data_ptr<float>(),input.data_ptr<scalar_t>(),
  hidden_size,d1);
  });
  break;

  case 4:
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_weight_kernel_d1", [&] {
  vllm::moe::moe_sum_weight_kernel_d1<scalar_t, 4><<<grid, block, 0, stream>>>(
  output.data_ptr<scalar_t>(), weight.data_ptr<float>(),input.data_ptr<scalar_t>(),
  hidden_size,d1);
  });
  break;

  case 5:
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_weight_kernel_d1", [&] {
  vllm::moe::moe_sum_weight_kernel_d1<scalar_t, 5><<<grid, block, 0, stream>>>(
  output.data_ptr<scalar_t>(), weight.data_ptr<float>(),input.data_ptr<scalar_t>(),
  hidden_size,d1);
  });
  break;

  case 6:
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_weight_kernel_d1", [&] {
  vllm::moe::moe_sum_weight_kernel_d1<scalar_t, 6><<<grid, block, 0, stream>>>(
  output.data_ptr<scalar_t>(), weight.data_ptr<float>(),input.data_ptr<scalar_t>(),
  hidden_size,d1);
  });
  break;

  case 7:
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_weight_kernel_d1", [&] {
  vllm::moe::moe_sum_weight_kernel_d1<scalar_t, 7><<<grid, block, 0, stream>>>(
  output.data_ptr<scalar_t>(), weight.data_ptr<float>(),input.data_ptr<scalar_t>(),
  hidden_size,d1);
  });
  break;

  case 8:
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_weight_kernel_d1", [&] {
  vllm::moe::moe_sum_weight_kernel_d1<scalar_t, 8><<<grid, block, 0, stream>>>(
  output.data_ptr<scalar_t>(), weight.data_ptr<float>(),input.data_ptr<scalar_t>(),
  hidden_size,d1);
  });
  break;

  default:
  at::sum_out(output, input, 1);
  break;
  }
}

void moe_weight_sum_v1(torch::Tensor& input,   // [num_tokens, topk, hidden_size]
  torch::Tensor& weight,  // [nt, topk]
  torch::Tensor& output,  // [num_tokens, hidden_size]
  torch::Tensor& flex_topk_dev,
  int64_t max_topk,
  int64_t num_tokens
) 
{
  const int hidden_size = input.size(-1);
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_weight_kernel_d1_dyn", [&] {
    vllm::moe::moe_sum_weight_kernel_d1_dyn<scalar_t><<<grid, block, 0, stream>>>(
        output.data_ptr<scalar_t>(),
        weight.data_ptr<float>(),
        input.data_ptr<scalar_t>(),
        hidden_size,
        (int)max_topk,
        flex_topk_dev.data_ptr<int>(),
        (int)num_tokens);
  });
}

void dynamic_add(torch::Tensor& input1,
    torch::Tensor& input2,
    torch::Tensor& flex_topk_dev
){
  // 若k == 0 ，则直接返回
  // 反之，将input2 加到 input1上
  const int hidden_size = input1.size(-1);
  const int num_tokens  = input1.size(0);
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input1));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "dynamic_add_kernel", [&] {
    vllm::moe::dynamic_add_kernel<scalar_t><<<grid, block, 0, stream>>>(
        input1.data_ptr<scalar_t>(),
        input2.data_ptr<scalar_t>(),
        flex_topk_dev.data_ptr<int>(),
        (int)num_tokens,
        (int)hidden_size);
  });
}

void dynamic_threshold(torch::Tensor& topk_weight,
                       torch::Tensor& alpha,
                       torch::Tensor& threshold,
                       torch::Tensor& flex_topk_dev,
                       torch::Tensor& flex_idx_dev,
                       torch::Tensor& bsz_tensor) {
  TORCH_CHECK(topk_weight.is_cuda(), "topk_weight must be a CUDA tensor");
  TORCH_CHECK(alpha.is_cuda(), "alpha must be a CUDA tensor");
  TORCH_CHECK(threshold.is_cuda(), "threshold must be a CUDA tensor");
  TORCH_CHECK(flex_topk_dev.is_cuda(), "flex_topk_dev must be a CUDA tensor");
  TORCH_CHECK(flex_idx_dev.is_cuda(), "flex_idx_dev must be a CUDA tensor");
  TORCH_CHECK(topk_weight.scalar_type() == torch::kFloat32, "topk_weight must be float32");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat32, "alpha must be float32");
  TORCH_CHECK(threshold.scalar_type() == torch::kFloat32, "threshold must be float32");
  TORCH_CHECK(flex_topk_dev.scalar_type() == torch::kInt32, "flex_topk_dev must be int32");
  TORCH_CHECK(flex_idx_dev.scalar_type() == torch::kInt32, "flex_idx_dev must be int32");
  TORCH_CHECK(topk_weight.dim() == 2, "topk_weight must be shape [num_tokens, num_experts]");
  TORCH_CHECK(topk_weight.size(1) >= 4, "topk_weight second dim must be >= 4");
  TORCH_CHECK(alpha.numel() == 1, "alpha must contain one scalar");
  TORCH_CHECK(threshold.numel() == 1, "threshold must contain one scalar");
  TORCH_CHECK(flex_topk_dev.numel() == 1, "flex_topk_dev must contain one scalar");
  TORCH_CHECK(flex_idx_dev.numel() == 1, "flex_idx_dev must contain one scalar");

  const int num_tokens = static_cast<int>(topk_weight.size(0));
  const int num_experts = static_cast<int>(topk_weight.size(1));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_weight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto avg_weight = torch::zeros({num_experts}, topk_weight.options());
  vllm::moe::dynamic_threshold_reduce_kernel<<<1, num_experts, 0, stream>>>(
      topk_weight.data_ptr<float>(),
      avg_weight.data_ptr<float>(),
      bsz_tensor.data_ptr<int>(),
      num_experts);
  vllm::moe::dynamic_threshold_decide_kernel<<<1, 1, 0, stream>>>(
      avg_weight.data_ptr<float>(),
      alpha.data_ptr<float>(),
      threshold.data_ptr<float>(),
      flex_topk_dev.data_ptr<int>(),
      flex_idx_dev.data_ptr<int>());
}