#include <cstdint>

template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void moe_q(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k) {
  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / QK8_1;
  const int blocks_per_warp = WARP_SIZE_GGUF / qi;

  const int ncols_dst = ncols_y * top_k;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int& row_x_0 = row_dst_0;

  const int col_dst_0 = blockIdx.y * mmq_x;

  int token_offs[mmq_x / nwarps];
  for (int i = 0; i < mmq_x; i += nwarps) {
    token_offs[i / nwarps] = sorted_token_ids[col_dst_0 + threadIdx.y + i];
  }

  const int exp_idx = expert_ids[blockIdx.y];
  if (exp_idx > 255 || exp_idx < 0) return;
  if (blockIdx.y * mmq_x > num_tokens_post_padded[0]) return;

  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_1* y = (const block_q8_1*)(vy);

  int* tile_x_ql = nullptr;
  half2* tile_x_dm = nullptr;
  int* tile_x_qh = nullptr;
  int* tile_x_sc = nullptr;

  allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

  __shared__ int tile_y_qs[mmq_x * WARP_SIZE_GGUF];
  __shared__ half2 tile_y_ds[mmq_x * WARP_SIZE_GGUF / QI8_1];

  float sum[mmq_y / WARP_SIZE_GGUF][mmq_x / nwarps] = {{0.0f}};

  for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {
    load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
               tile_x_qh, tile_x_sc, threadIdx.y, nrows_x - row_x_0 - 1,
               threadIdx.x, blocks_per_row_x);

    const int n_per_r = ((qk * blocks_per_warp) / qr);
#pragma unroll
    for (int ir = 0; ir < qr && ib0 * qk + ir * n_per_r < ncols_x; ++ir) {
      const int kqs = ir * WARP_SIZE_GGUF + threadIdx.x;
      const int kbxd = kqs / QI8_1;

#pragma unroll
      for (int i = 0; i < mmq_x; i += nwarps) {
        const int col_y_eff = token_offs[i / nwarps] / top_k;
        const int block_x = ib0 * (qk / QK8_1) + kbxd;
        if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
        const block_q8_1* by0 = &y[col_y_eff * blocks_per_col_y + block_x];
        const int index_y =
            (threadIdx.y + i) * WARP_SIZE_GGUF + kqs % WARP_SIZE_GGUF;
          tile_y_qs[index_y] =
              get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
      }
      }

      if (threadIdx.x < n_per_r / QK8_1) {
        const int kby = threadIdx.x % (WARP_SIZE_GGUF / QI8_1);
        const int col_y_eff = token_offs[threadIdx.y] / top_k;
        const int block_x =
            ib0 * (qk / QK8_1) + ir * (WARP_SIZE_GGUF / QI8_1) + kby;

        if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
        const half2* dsi_src = &y[col_y_eff * blocks_per_col_y + block_x].ds;
        half2* dsi_dst =
            &tile_y_ds[threadIdx.y * (WARP_SIZE_GGUF / QI8_1) + kby];

        if (need_sum) {
            *dsi_dst = *dsi_src;
        } else {
          float* dfi_dst = (float*)dsi_dst;
            *dfi_dst = __low2float(*dsi_src);
          }
        }
      }
      __syncthreads();

      // #pragma unroll // unrolling this loop causes too much register pressure
      for (int k = ir * WARP_SIZE_GGUF / qr; k < (ir + 1) * WARP_SIZE_GGUF / qr;
           k += vdr) {
#pragma unroll
        for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
          for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
            sum[i / WARP_SIZE_GGUF][j / nwarps] +=
                vec_dot(tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs,
                        tile_y_ds, threadIdx.x + i, threadIdx.y + j, k);
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int j = 0; j < mmq_x; j += nwarps) {
    const int col_dst = token_offs[j / nwarps];
    if (col_dst >= ncols_dst) {
      return;
    }

#pragma unroll
    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
      const int row_dst = row_dst_0 + threadIdx.x + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE_GGUF][j / nwarps];
    }
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q4_0 64
  #define MMQ_Y_Q4_0 128
  #define NWARPS_Q4_0 8
#else
  #define MMQ_X_Q4_0 4
  #define MMQ_Y_Q4_0 32
  #define NWARPS_Q4_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_0, 2)
#endif
    moe_q4_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MMQ_X_Q4_0;
  const int mmq_y = MMQ_Y_Q4_0;
  const int nwarps = NWARPS_Q4_0;

  moe_q<scalar_t, QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
        VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  int mmq_x = MMQ_X_Q4_0;
  int mmq_y = MMQ_Y_Q4_0;
  int nwarps = NWARPS_Q4_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q4_1 64
  #define MMQ_Y_Q4_1 128
  #define NWARPS_Q4_1 8
#else
  #define MMQ_X_Q4_1 4
  #define MMQ_Y_Q4_1 32
  #define NWARPS_Q4_1 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_1, 2)
#endif
    moe_q4_1(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MMQ_X_Q4_1;
  const int mmq_y = MMQ_Y_Q4_1;
  const int nwarps = NWARPS_Q4_1;

  moe_q<scalar_t, QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_1<mmq_y>, load_tiles_q4_1<mmq_y, nwarps, need_check>,
        VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_1_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  int mmq_x = MMQ_X_Q4_1;
  int mmq_y = MMQ_Y_Q4_1;
  int nwarps = NWARPS_Q4_1;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q5_0 64
  #define MMQ_Y_Q5_0 128
  #define NWARPS_Q5_0 8
#else
  #define MMQ_X_Q5_0 4
  #define MMQ_Y_Q5_0 32
  #define NWARPS_Q5_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_0, 2)
#endif
    moe_q5_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MMQ_X_Q5_0;
  const int mmq_y = MMQ_Y_Q5_0;
  const int nwarps = NWARPS_Q5_0;

  moe_q<scalar_t, QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_0<mmq_y>, load_tiles_q5_0<mmq_y, nwarps, need_check>,
        VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MMQ_X_Q5_0;
  const int mmq_y = MMQ_Y_Q5_0;
  const int nwarps = NWARPS_Q5_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q5_1 64
  #define MMQ_Y_Q5_1 128
  #define NWARPS_Q5_1 8
#else
  #define MMQ_X_Q5_1 4
  #define MMQ_Y_Q5_1 32
  #define NWARPS_Q5_1 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_1, 2)
#endif
    moe_q5_1(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MMQ_X_Q5_1;
  const int mmq_y = MMQ_Y_Q5_1;
  const int nwarps = NWARPS_Q5_1;

  moe_q<scalar_t, QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_1<mmq_y>, load_tiles_q5_1<mmq_y, nwarps, need_check>,
        VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_1_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MMQ_X_Q5_1;
  const int mmq_y = MMQ_Y_Q5_1;
  const int nwarps = NWARPS_Q5_1;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q8_0 64
  #define MMQ_Y_Q8_0 128
  #define NWARPS_Q8_0 8
#else
  #define MMQ_X_Q8_0 4
  #define MMQ_Y_Q8_0 32
  #define NWARPS_Q8_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q8_0, 2)
#endif
    moe_q8_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MMQ_X_Q8_0;
  const int mmq_y = MMQ_Y_Q8_0;
  const int nwarps = NWARPS_Q8_0;

  moe_q<scalar_t, QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q8_0<mmq_y>, load_tiles_q8_0<mmq_y, nwarps, need_check>,
        VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q8_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MMQ_X_Q8_0;
  const int mmq_y = MMQ_Y_Q8_0;
  const int nwarps = NWARPS_Q8_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q2_K 64
  #define MMQ_Y_Q2_K 128
  #define NWARPS_Q2_K 8
#else
  #define MMQ_X_Q2_K 4
  #define MMQ_Y_Q2_K 32
  #define NWARPS_Q2_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q2_K, 2)
#endif
    moe_q2_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MMQ_X_Q2_K;
  const int mmq_y = MMQ_Y_Q2_K;
  const int nwarps = NWARPS_Q2_K;

  moe_q<scalar_t, QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q2_K<mmq_y>, load_tiles_q2_K<mmq_y, nwarps, need_check>,
        VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q2_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MMQ_X_Q2_K;
  const int mmq_y = MMQ_Y_Q2_K;
  const int nwarps = NWARPS_Q2_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q2_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q3_K 64
  #define MMQ_Y_Q3_K 128
  #define NWARPS_Q3_K 8
#else
  #define MMQ_X_Q3_K 4
  #define MMQ_Y_Q3_K 32
  #define NWARPS_Q3_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q3_K, 2)
#endif
    moe_q3_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {

  const int mmq_x = MMQ_X_Q3_K;
  const int mmq_y = MMQ_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  moe_q<scalar_t, QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q3_K<mmq_y>, load_tiles_q3_K<mmq_y, nwarps, need_check>,
        VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}
template <typename scalar_t>
static void ggml_moe_q3_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MMQ_X_Q3_K;
  const int mmq_y = MMQ_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q4_K 64
  #define MMQ_Y_Q4_K 128
  #define NWARPS_Q4_K 8
#else
  #define MMQ_X_Q4_K 4
  #define MMQ_Y_Q4_K 32
  #define NWARPS_Q4_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_K, 2)
#endif
    moe_q4_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MMQ_X_Q4_K;
  const int mmq_y = MMQ_Y_Q4_K;
  const int nwarps = NWARPS_Q4_K;

  moe_q<scalar_t, QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_K<mmq_y>, load_tiles_q4_K<mmq_y, nwarps, need_check>,
        VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MMQ_X_Q4_K;
  const int mmq_y = MMQ_Y_Q4_K;
  const int nwarps = NWARPS_Q4_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q5_K 64
  #define MMQ_Y_Q5_K 128
  #define NWARPS_Q5_K 8
#else
  #define MMQ_X_Q5_K 4
  #define MMQ_Y_Q5_K 32
  #define NWARPS_Q5_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_K, 2)
#endif
    moe_q5_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MMQ_X_Q5_K;
  const int mmq_y = MMQ_Y_Q5_K;
  const int nwarps = NWARPS_Q5_K;

  moe_q<scalar_t, QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_K<mmq_y>, load_tiles_q5_K<mmq_y, nwarps, need_check>,
        VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MMQ_X_Q5_K;
  const int mmq_y = MMQ_Y_Q5_K;
  const int nwarps = NWARPS_Q5_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MMQ_X_Q6_K 64
  #define MMQ_Y_Q6_K 128
  #define NWARPS_Q6_K 8
#else
  #define MMQ_X_Q6_K 4
  #define MMQ_Y_Q6_K 32
  #define NWARPS_Q6_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q6_K, 2)
#endif
    moe_q6_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MMQ_X_Q6_K;
  const int mmq_y = MMQ_Y_Q6_K;
  const int nwarps = NWARPS_Q6_K;

  moe_q<scalar_t, QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q6_K<mmq_y>, load_tiles_q6_K<mmq_y, nwarps, need_check>,
        VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q6_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MMQ_X_Q6_K;
  const int mmq_y = MMQ_Y_Q6_K;
  const int nwarps = NWARPS_Q6_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}


template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_r4_shared_cuda_t vec_dot>
static __device__ __forceinline__ void moe_vec_q_shared(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k) {
  const int blocks_per_row_x = ncols_x / qk;
  // Q8_K128 每个block处理128
  const int blocks_per_col_y = nrows_y / 128;
  const int row_size = 2 + ncols_x * 1.5 / 8;

  const int ncols_dst = ncols_y;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int row = row_dst_0 + threadIdx.x + WARP_SIZE_GGUF * threadIdx.y;

  const int col_dst_0 = blockIdx.y * mmq_x;
  // 这里默认mmq_x = 1
  const int token_ids = sorted_token_ids[col_dst_0];
  const int token_offs = token_ids / top_k;

  const int exp_idx = expert_ids[blockIdx.y];
  if (exp_idx > 255 || exp_idx < 0) return;
  if (blockIdx.y * mmq_x > num_tokens_post_padded[0]) return;

  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_K128* y = (const block_q8_K128*)(vy);

  float sum[mmq_y / (WARP_SIZE_GGUF*nwarps)] = {0.0f};
  // v1:不采用shared memory
  const int col_y_eff = token_offs;
  const block_q8_K128* yy = (const block_q8_K128*)(y + col_y_eff * blocks_per_col_y);

  __shared__ uint64_t shared_iq1s_grid[2048];

  // 计算线性线程ID，考虑x和y维度
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y; // 总线程数

  // 使用ulonglong2向量加载（每个ulonglong2包含2个uint64_t）
  const int vec2_elements = 2048 / 2;
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;

  // 将uint64_t数组视为ulonglong2数组进行加载
  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_gpu);
  ulonglong2* shared_iq1s_grid_vec2 = reinterpret_cast<ulonglong2*>(shared_iq1s_grid);

  for (int i = 0; i < vec2_per_thread; i++) {
      int idx = tid + i * block_size;
      if (idx < vec2_elements) {
          shared_iq1s_grid_vec2[idx] = __ldg(&iq1s_grid_vec2[idx]);
      }
  }

  // 确保所有线程都完成加载
  __syncthreads();

#pragma unroll
  for(int k = 0; k < mmq_y; k += WARP_SIZE_GGUF*nwarps){
    float d;
    const int row_offs = (row + k) / 4;
    const int iqs = threadIdx.x % 4;
    const half * df = (const half *) ((const char *)x + (row_offs) * row_size * 4);
    d = __half2float(df[iqs]);
    const block_q_t* xx = (const block_q_t*)(df+4);
    #pragma unroll
    for (int i = 0; i < blocks_per_row_x; i += 1) {
      const int ibx = i;
      const int iby = (ibx * qk/(128));
      const int iqa = i % 4;
      sum[k/(WARP_SIZE_GGUF*nwarps)] += vec_dot(&xx[ibx], &yy[iby], iqs, iqa, d, shared_iq1s_grid);
    }
  }


#pragma unroll
    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF*nwarps) {
      const int row_dst = row_dst_0 + threadIdx.x +  WARP_SIZE_GGUF * threadIdx.y + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      dst[col_dst_0 * nrows_dst + row_dst] = sum[i / (WARP_SIZE_GGUF*nwarps)];
  }
}


template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_r4_shared_cuda_t vec_dot>
static __device__ __forceinline__ void moe_vec_q_shared_1(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k) {
  const int reduce_per_warp = qi;
  const int row_per_warp = WARP_SIZE_GGUF / reduce_per_warp;
  
  const int blocks_per_row_x = ncols_x / qk;
  // Q8_K128 每个block处理128
  const int blocks_per_col_y = nrows_y / 128;
  const int row_size = std::is_same<block_q_t, block_iq1_s_r4>::value ?  (2 + ncols_x * 1.5 / 8) :
                       std::is_same<block_q_t, block_iq1_m_r4>::value ? 
                       (2 + ncols_x * 7 / 32) : 0;

  const int ncols_dst = ncols_y;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int row = row_dst_0 + threadIdx.x % row_per_warp + row_per_warp * threadIdx.y;

  const int col_dst_0 = blockIdx.y * mmq_x;
  // 这里默认mmq_x = 1
  const int token_ids = sorted_token_ids[col_dst_0];
  const int token_offs = token_ids / top_k;

  const int exp_idx = expert_ids[blockIdx.y];
  if (exp_idx > 255 || exp_idx < 0) return;
  if (blockIdx.y * mmq_x > num_tokens_post_padded[0]) return;

  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  //printf("BlockIdx y: %d, exp_idx: %d",blockIdx.y,exp_idx);
  const block_q8_K128* y = (const block_q8_K128*)(vy);

  float sum[mmq_y / (row_per_warp*nwarps)] = {0.0f};
  const int col_y_eff = token_offs;
  const block_q8_K128* yy = (const block_q8_K128*)(y + col_y_eff * blocks_per_col_y);

  __shared__ uint64_t shared_iq1s_grid[2048];

  // 计算线性线程ID，考虑x和y维度
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y; // 总线程数

  // 使用ulonglong2向量加载（每个ulonglong2包含2个uint64_t）
  const int vec2_elements = 2048 / 2;
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;

  // 将uint64_t数组视为ulonglong2数组进行加载
  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_gpu);
  ulonglong2* shared_iq1s_grid_vec2 = reinterpret_cast<ulonglong2*>(shared_iq1s_grid);
  #pragma unroll
  for (int i = 0; i < vec2_per_thread; i++) {
      int idx = tid + i * block_size;
      if (idx < vec2_elements) {
          shared_iq1s_grid_vec2[idx] = __ldg(&iq1s_grid_vec2[idx]);
      }
  }

  // 确保所有线程都完成加载
  __syncthreads();

#pragma unroll
  for(int k = 0; k < mmq_y; k += row_per_warp*nwarps){
    float d;
    const int row_offs = (row + k) / 4;
    const int iqs = threadIdx.x % 4;
    const half * df = (const half *) ((const char *)x + (row_offs) * row_size * 4);
    d = __half2float(df[iqs]);
    const block_q_t* xx = (const block_q_t*)(df+4);
    #pragma unroll
    for (int i = 0; i < blocks_per_row_x; i += reduce_per_warp) {
      const int ibx = i + threadIdx.x / row_per_warp;
      const int iby = (ibx * qk/(128));
      const int iqa = ibx % 4;
      sum[k/(row_per_warp*nwarps)] += vec_dot(&xx[ibx], &yy[iby], iqs, iqa, d, shared_iq1s_grid);
    }
  }
#pragma unroll
// 

#pragma unroll
    for (int i = 0; i < mmq_y; i += row_per_warp*nwarps) {
      const int row_dst = row_dst_0 + threadIdx.x % row_per_warp +  row_per_warp * threadIdx.y + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      #pragma unroll
      for (int mask = 16; mask >= row_per_warp; mask >>= 1){
        sum[i / (row_per_warp*nwarps)] += VLLM_SHFL_XOR_SYNC(sum[i / (row_per_warp*nwarps)], mask);
      }
      if(threadIdx.x / row_per_warp == 0){
        dst[col_dst_0 * nrows_dst + row_dst] = sum[i / (row_per_warp*nwarps)];
      }
  }
}

template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_r4_shared_cuda_t vec_dot>
static __device__ __forceinline__ void moe_vec_q_shared_2(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k) {
  // shared_2 对比 shared_1, 改变在于固定nwarps为4，R4中的连续四行，从而支持warp内高达32的K并行
  const int reduce_per_warp = qi;
  const int row_per_warp = WARP_SIZE_GGUF / reduce_per_warp;
  
  const int blocks_per_row_x = ncols_x / qk;
  // Q8_K128 每个block处理128
  const int blocks_per_col_y = nrows_y / 128;
  const int row_size = 2 + ncols_x * 1.5 / 8;

  const int ncols_dst = ncols_y;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int row = row_dst_0 + threadIdx.x % row_per_warp * 4 + threadIdx.y;

  const int col_dst_0 = blockIdx.y * mmq_x;
  // 这里默认mmq_x = 1
  const int token_ids = sorted_token_ids[col_dst_0];
  const int token_offs = token_ids / top_k;

  const int exp_idx = expert_ids[blockIdx.y];
  if (exp_idx > 255 || exp_idx < 0) return;
  if (blockIdx.y * mmq_x > num_tokens_post_padded[0]) return;

  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_K128* y = (const block_q8_K128*)(vy);

  float sum[mmq_y / (row_per_warp*nwarps)] = {0.0f};
  const int col_y_eff = token_offs;
  const block_q8_K128* yy = (const block_q8_K128*)(y + col_y_eff * blocks_per_col_y);

  __shared__ uint64_t shared_iq1s_grid[2048];

  // 计算线性线程ID，考虑x和y维度
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y; // 总线程数

  // 使用ulonglong2向量加载（每个ulonglong2包含2个uint64_t）
  const int vec2_elements = 2048 / 2;
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;

  // 将uint64_t数组视为ulonglong2数组进行加载
  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_gpu);
  ulonglong2* shared_iq1s_grid_vec2 = reinterpret_cast<ulonglong2*>(shared_iq1s_grid);
  #pragma unroll
  for (int i = 0; i < vec2_per_thread; i++) {
      int idx = tid + i * block_size;
      if (idx < vec2_elements) {
          shared_iq1s_grid_vec2[idx] = __ldg(&iq1s_grid_vec2[idx]);
      }
  }

  // 确保所有线程都完成加载
  __syncthreads();

#pragma unroll
  for(int k = 0; k < mmq_y; k += row_per_warp*nwarps){
    float d;
    const int row_offs = (row + k) / 4;
    const int iqs = threadIdx.y % 4;
    const half * df = (const half *) ((const char *)x + (row_offs) * row_size * 4);
    d = __half2float(df[iqs]);
    const block_q_t* xx = (const block_q_t*)(df+4);
    #pragma unroll
    for (int i = 0; i < blocks_per_row_x; i += reduce_per_warp) {
      const int ibx = i + threadIdx.x / row_per_warp;
      const int iby = (ibx * qk/(128));
      const int iqa = ibx % 4;
      sum[k/(row_per_warp*nwarps)] += vec_dot(&xx[ibx], &yy[iby], iqs, iqa, d, shared_iq1s_grid);
    }
  }
#pragma unroll
// 

#pragma unroll
    for (int i = 0; i < mmq_y; i += row_per_warp*nwarps) {
      const int row_dst = row_dst_0 + threadIdx.x % row_per_warp * 4 + threadIdx.y + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      #pragma unroll
      for (int mask = 16; mask >= row_per_warp; mask >>= 1){
        sum[i / (row_per_warp*nwarps)] += VLLM_SHFL_XOR_SYNC(sum[i / (row_per_warp*nwarps)], mask);
      }
      if(threadIdx.x / row_per_warp == 0){
        dst[col_dst_0 * nrows_dst + row_dst] = sum[i / (row_per_warp*nwarps)];
      }
  }
}


template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_r4_cuda_t vec_dot>
static __device__ __forceinline__ void moe_vec_q(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k) {
  const int blocks_per_row_x = ncols_x / qk;
  // Q8_K128 每个block处理128
  const int blocks_per_col_y = nrows_y / 128;
  const int row_size = 2 + ncols_x * 1.5 / 8;

  const int ncols_dst = ncols_y;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int row = row_dst_0 + threadIdx.x + WARP_SIZE_GGUF * threadIdx.y;

  const int col_dst_0 = blockIdx.y * mmq_x;
  // 这里默认mmq_x = 1
  const int token_ids = sorted_token_ids[col_dst_0];
  const int token_offs = token_ids / top_k;

  const int exp_idx = expert_ids[blockIdx.y];
  if (exp_idx > 255 || exp_idx < 0) return;
  if (blockIdx.y * mmq_x > num_tokens_post_padded[0]) return;

  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_K128* y = (const block_q8_K128*)(vy);

  float sum[mmq_y / (WARP_SIZE_GGUF*nwarps)] = {0.0f};
  // v1:不采用shared memory
  const int col_y_eff = token_offs;
  const block_q8_K128* yy = (const block_q8_K128*)(y + col_y_eff * blocks_per_col_y);
#pragma unroll
  for(int k = 0; k < mmq_y; k += WARP_SIZE_GGUF*nwarps){
    float d;
    const int row_offs = (row + k) / 4;
    const int iqs = threadIdx.x % 4;
    const half * df = (const half *) ((const char *)x + (row_offs) * row_size * 4);
    d = __half2float(df[iqs]);
    const block_q_t* xx = (const block_q_t*)(df+4);
    #pragma unroll
    for (int i = 0; i < blocks_per_row_x; i += 1) {
      const int ibx = i;
      const int iby = (ibx * qk/(128));
      const int iqa = i % 4;
      sum[k/(WARP_SIZE_GGUF*nwarps)] += vec_dot(&xx[ibx], &yy[iby], iqs, iqa, d);
    }
  }


#pragma unroll
    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF*nwarps) {
      const int row_dst = row_dst_0 + threadIdx.x +  WARP_SIZE_GGUF * threadIdx.y + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      dst[col_dst_0 * nrows_dst + row_dst] = sum[i / (WARP_SIZE_GGUF*nwarps)];
  }
}
#if defined(USE_ROCM)
  #define MMQ_X_IQ1_R4 64
  #define MMQ_Y_IQ1_R4 128
  #define NWARPS_IQ1_R4 8
#else
  #define MMQ_X_IQ1_R4 1
  #define MMQ_Y_IQ1_R4 32
  #define NWARPS_IQ1_R4 8
#endif

#ifndef MMQ_X_IQ1_M_R4_PREFILL_BATCH
#define MMQ_X_IQ1_M_R4_PREFILL_BATCH 1
#endif

#if MMQ_X_IQ1_M_R4_PREFILL_BATCH != 1 && MMQ_X_IQ1_M_R4_PREFILL_BATCH != 2 && \
    MMQ_X_IQ1_M_R4_PREFILL_BATCH != 4 && MMQ_X_IQ1_M_R4_PREFILL_BATCH != 8
#error "MMQ_X_IQ1_M_R4_PREFILL_BATCH must be one of {1,2,4,8}"
#endif
// 每个BLock负责计算1*32的子矩阵

// Graph-capture friendly variant:
// - Do NOT require host-side slicing/compaction.
// - Read idx/topk/tokens/expert_slice_start from 1-element CUDA tensors on device.
// - Compute expert_id from (token_id, rank) directly from the full expert table.
template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_r4_shared_cuda_t vec_dot>
static __device__ __forceinline__ void moe_vec_q_shared_1_dyn(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst,
    const int max_topk) {
  const int reduce_per_warp = qi;
  const int row_per_warp = WARP_SIZE_GGUF / reduce_per_warp;

  const int blocks_per_row_x = ncols_x / qk;
  // Q8_K128 每个block处理128
  const int blocks_per_col_y = nrows_y / 128;
  const int row_size = std::is_same<block_q_t, block_iq1_s_r4>::value ?  (2 + ncols_x * 1.5 / 8) :
                       std::is_same<block_q_t, block_iq1_m_r4>::value ?
                       (2 + ncols_x * 7 / 32) : 0;

  const int ncols_dst = ncols_y;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int row = row_dst_0 + threadIdx.x % row_per_warp + row_per_warp * threadIdx.y;

  const int idx = __ldg(idx_dev);
  const int top_k = __ldg(topk_dev);
  const int sorted_slice_start = __ldg(sorted_slice_start_dev);
  const int expert_slice_start = __ldg(expert_slice_start_dev);

  // Replicate host-side slice_start logic (must stay deterministic for capture).
  int slice_start = sorted_slice_start;
  int expert_ids_offs = blockIdx.y / top_k;
  int expert_ids_idx  = blockIdx.y % top_k;

  const int col_dst_0 = blockIdx.y * mmq_x;
  const int limit = num_tokens_post_padded_all[idx];
  if (blockIdx.y * mmq_x >= limit) return;
  // 这里默认mmq_x = 1
  const int token_ids = sorted_token_ids_all[slice_start + col_dst_0];
  const int token_offs = token_ids / top_k;

  // Compute expert id from full table: [token_num, max_topk]
  const int exp_idx = expert_ids_full[expert_slice_start + expert_ids_offs * max_topk + expert_ids_idx];
  if (exp_idx > 255 || exp_idx < 0) return;


  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_K128* y = (const block_q8_K128*)(vy);

  float sum[mmq_y / (row_per_warp*nwarps)] = {0.0f};
  const int col_y_eff = token_offs;
  const block_q8_K128* yy = (const block_q8_K128*)(y + col_y_eff * blocks_per_col_y);

  __shared__ uint64_t shared_iq1s_grid[2048];

  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;

  const int vec2_elements = 2048 / 2;
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;

  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_gpu);
  ulonglong2* shared_iq1s_grid_vec2 = reinterpret_cast<ulonglong2*>(shared_iq1s_grid);
  #pragma unroll
  for (int i = 0; i < vec2_per_thread; i++) {
      int j = tid + i * block_size;
      if (j < vec2_elements) {
          shared_iq1s_grid_vec2[j] = __ldg(&iq1s_grid_vec2[j]);
      }
  }

  __syncthreads();

#pragma unroll
  for(int k = 0; k < mmq_y; k += row_per_warp*nwarps){
    float d;
    const int row_offs = (row + k) / 4;
    const int iqs = threadIdx.x % 4;
    const half * df = (const half *) ((const char *)x + (row_offs) * row_size * 4);
    d = __half2float(df[iqs]);
    const block_q_t* xx = (const block_q_t*)(df+4);
    #pragma unroll
    for (int i = 0; i < blocks_per_row_x; i += reduce_per_warp) {
      const int ibx = i + threadIdx.x / row_per_warp;
      const int iby = (ibx * qk/(128));
      const int iqa = ibx % 4;
      sum[k/(row_per_warp*nwarps)] += vec_dot(&xx[ibx], &yy[iby], iqs, iqa, d, shared_iq1s_grid);
    }
  }

#pragma unroll
  for (int i = 0; i < mmq_y; i += row_per_warp*nwarps) {
    const int row_dst = row_dst_0 + threadIdx.x % row_per_warp +  row_per_warp * threadIdx.y + i;
    if (row_dst >= nrows_dst) {
      continue;
    }
    #pragma unroll
    for (int mask = 16; mask >= row_per_warp; mask >>= 1){
      sum[i / (row_per_warp*nwarps)] += VLLM_SHFL_XOR_SYNC(sum[i / (row_per_warp*nwarps)], mask);
    }
    if(threadIdx.x / row_per_warp == 0){
      dst[col_dst_0 * nrows_dst + row_dst] = sum[i / (row_per_warp*nwarps)];
    }
  }
}


// Graph-capture friendly variant for moe_gemm_w2:
// - topk_dev stores flex_topk (slice start)
// - top_k is passed as a host constant (matches token_ids encoding)
template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_r4_shared_cuda_t vec_dot>
static __device__ __forceinline__ void moe_vec_q_shared_1_dyn_w2(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ flex_topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst,
    const int top_k,
    const int max_topk) {
  const int reduce_per_warp = qi;
  const int row_per_warp = WARP_SIZE_GGUF / reduce_per_warp;

  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / 128;
  const int row_size = std::is_same<block_q_t, block_iq1_s_r4>::value ?  (2 + ncols_x * 1.5 / 8) :
                       std::is_same<block_q_t, block_iq1_m_r4>::value ?
                       (2 + ncols_x * 7 / 32) : 0;

  const int ncols_dst = ncols_y;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int row = row_dst_0 + threadIdx.x % row_per_warp + row_per_warp * threadIdx.y;

  const int idx = __ldg(idx_dev);
  const int flex_topk = __ldg(flex_topk_dev);
  const int sorted_slice_start = __ldg(sorted_slice_start_dev);
  const int expert_slice_start = flex_topk;

  const int col_dst_0 = blockIdx.y * mmq_x;
  const int limit = num_tokens_post_padded_all[idx];
  if (blockIdx.y * mmq_x >= limit) return;

  const int token_ids = sorted_token_ids_all[sorted_slice_start + col_dst_0];
  const int token_offs = token_ids / top_k;
  int expert_ids_offs = blockIdx.y / (max_topk - flex_topk);
  int expert_ids_idx  = blockIdx.y % (max_topk - flex_topk);
  
  const int exp_idx = expert_ids_full[expert_slice_start + expert_ids_offs * max_topk + expert_ids_idx];
  //printf("col_dst_0: %d, token_ids: %d, token_offs: %d, exp_idx: %d\n", col_dst_0, token_ids, token_offs, exp_idx);
  if (exp_idx > 255 || exp_idx < 0) return;


  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_K128* y = (const block_q8_K128*)(vy);

  float sum[mmq_y / (row_per_warp*nwarps)] = {0.0f};
  const int col_y_eff = token_offs;
  const block_q8_K128* yy = (const block_q8_K128*)(y + col_y_eff * blocks_per_col_y);

  __shared__ uint64_t shared_iq1s_grid[2048];

  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;

  const int vec2_elements = 2048 / 2;
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;

  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_gpu);
  ulonglong2* shared_iq1s_grid_vec2 = reinterpret_cast<ulonglong2*>(shared_iq1s_grid);
  #pragma unroll
  for (int i = 0; i < vec2_per_thread; i++) {
      int j = tid + i * block_size;
      if (j < vec2_elements) {
          shared_iq1s_grid_vec2[j] = __ldg(&iq1s_grid_vec2[j]);
      }
  }

  __syncthreads();

  for (int k = 0; k < mmq_y; k += row_per_warp*nwarps) {
    const int row_offs = (row + k) / 4;
    const int iqs = threadIdx.x % 4;
    float d = 1.0f;
    const half * df = (const half *) ((const char *)x + (row_offs) * row_size * 4);
    d = __half2float(df[iqs]);
    const block_q_t* xx = (const block_q_t*)(df+4);
    #pragma unroll
    for (int i = 0; i < blocks_per_row_x; i += reduce_per_warp) {
      const int ibx = i + threadIdx.x / row_per_warp;
      const int iby = (ibx * qk/(128));
      const int iqa = ibx % 4;
      sum[k/(row_per_warp*nwarps)] += vec_dot(&xx[ibx], &yy[iby], iqs, iqa, d, shared_iq1s_grid);
    }
  }

  #pragma unroll
  for (int i = 0; i < mmq_y; i += row_per_warp*nwarps) {
    const int row_dst = row_dst_0 + threadIdx.x % row_per_warp +  row_per_warp * threadIdx.y + i;
    if (row_dst >= nrows_dst) {
      continue;
    }
    #pragma unroll
    for (int mask = 16; mask >= row_per_warp; mask >>= 1){
      sum[i / (row_per_warp*nwarps)] += VLLM_SHFL_XOR_SYNC(sum[i / (row_per_warp*nwarps)], mask);
    }
    if(threadIdx.x / row_per_warp == 0){
      dst[col_dst_0 * nrows_dst + row_dst] = sum[i / (row_per_warp*nwarps)];
    }
  }
}

// Graph-capture friendly variant for moe_gemm_w2:
// - topk_dev stores flex_topk (slice start)
// - top_k is passed as a host constant (matches token_ids encoding)
template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_r4_shared_cuda_t vec_dot>
static __device__ __forceinline__ void moe_vec_q_shared_1_dyn_w2_prefill(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst,
    const int top_k,
    const int max_topk) {
  const int reduce_per_warp = qi;
  const int row_per_warp = WARP_SIZE_GGUF / reduce_per_warp;

  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / 128;
  const int row_size = std::is_same<block_q_t, block_iq1_s_r4>::value ?  (2 + ncols_x * 1.5 / 8) :
                       std::is_same<block_q_t, block_iq1_m_r4>::value ?
                       (2 + ncols_x * 7 / 32) : 0;

  const int ncols_dst = ncols_y;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int row = row_dst_0 + threadIdx.x % row_per_warp + row_per_warp * threadIdx.y;

  const int idx = __ldg(idx_dev);
  const int sorted_slice_start = __ldg(sorted_slice_start_dev);
  const int expert_slice_start = __ldg(expert_slice_start_dev);
  const int limit = num_tokens_post_padded_all[idx];
  if (blockIdx.y * mmq_x >= limit) return;
  const int col_dst_0 = blockIdx.y * mmq_x;
  const int token_ids = sorted_token_ids_all[sorted_slice_start + col_dst_0];
  const int token_offs = token_ids / top_k;
  
  const int exp_idx = expert_ids_full[expert_slice_start + blockIdx.y];
  //printf("col_dst_0: %d, token_ids: %d, token_offs: %d, exp_idx: %d\n", col_dst_0, token_ids, token_offs, exp_idx);
  if (exp_idx > 255 || exp_idx < 0) return;


  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_K128* y = (const block_q8_K128*)(vy);

  float sum[mmq_y / (row_per_warp*nwarps)] = {0.0f};
  const int col_y_eff = token_offs;
  const block_q8_K128* yy = (const block_q8_K128*)(y + col_y_eff * blocks_per_col_y);

  __shared__ uint64_t shared_iq1s_grid[2048];

  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;

  const int vec2_elements = 2048 / 2;
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;

  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_gpu);
  ulonglong2* shared_iq1s_grid_vec2 = reinterpret_cast<ulonglong2*>(shared_iq1s_grid);
  #pragma unroll
  for (int i = 0; i < vec2_per_thread; i++) {
      int j = tid + i * block_size;
      if (j < vec2_elements) {
          shared_iq1s_grid_vec2[j] = __ldg(&iq1s_grid_vec2[j]);
      }
  }

  __syncthreads();

  for (int k = 0; k < mmq_y; k += row_per_warp*nwarps) {
    const int row_offs = (row + k) / 4;
    const int iqs = threadIdx.x % 4;
    float d = 1.0f;
    const half * df = (const half *) ((const char *)x + (row_offs) * row_size * 4);
    d = __half2float(df[iqs]);
    const block_q_t* xx = (const block_q_t*)(df+4);
    #pragma unroll
    for (int i = 0; i < blocks_per_row_x; i += reduce_per_warp) {
      const int ibx = i + threadIdx.x / row_per_warp;
      const int iby = (ibx * qk/(128));
      const int iqa = ibx % 4;
      sum[k/(row_per_warp*nwarps)] += vec_dot(&xx[ibx], &yy[iby], iqs, iqa, d, shared_iq1s_grid);
    }
  }

  #pragma unroll
  for (int i = 0; i < mmq_y; i += row_per_warp*nwarps) {
    const int row_dst = row_dst_0 + threadIdx.x % row_per_warp +  row_per_warp * threadIdx.y + i;
    if (row_dst >= nrows_dst) {
      continue;
    }
    #pragma unroll
    for (int mask = 16; mask >= row_per_warp; mask >>= 1){
      sum[i / (row_per_warp*nwarps)] += VLLM_SHFL_XOR_SYNC(sum[i / (row_per_warp*nwarps)], mask);
    }
    if(threadIdx.x / row_per_warp == 0){
      dst[col_dst_0 * nrows_dst + row_dst] = sum[i / (row_per_warp*nwarps)];
    }
  }
}


template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_r4_shared_cuda_t vec_dot>
static __device__ __forceinline__ void moe_vec_q_shared_1_dyn_prefill(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst,
    const int max_topk) {
  const int reduce_per_warp = qi;
  const int row_per_warp = WARP_SIZE_GGUF / reduce_per_warp;

  const int blocks_per_row_x = ncols_x / qk;
  // Q8_K128 每个block处理128
  const int blocks_per_col_y = nrows_y / 128;
  const int row_size = std::is_same<block_q_t, block_iq1_s_r4>::value ?  (2 + ncols_x * 1.5 / 8) :
                       std::is_same<block_q_t, block_iq1_m_r4>::value ?
                       (2 + ncols_x * 7 / 32) : 0;

  const int ncols_dst = ncols_y;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int row = row_dst_0 + threadIdx.x % row_per_warp + row_per_warp * threadIdx.y;

  const int idx = __ldg(idx_dev);
  const int top_k = __ldg(topk_dev);
  const int sorted_slice_start = __ldg(sorted_slice_start_dev);
  const int expert_slice_start = __ldg(expert_slice_start_dev);

  // Replicate host-side slice_start logic (must stay deterministic for capture).

  const int col_dst_0 = blockIdx.y * mmq_x;
  const int limit = num_tokens_post_padded_all[idx];
  if (blockIdx.y * mmq_x >= limit) return;
  // 这里默认mmq_x = 1
  const int token_ids = sorted_token_ids_all[sorted_slice_start + col_dst_0];
  const int token_offs = token_ids / top_k;

  // Compute expert id from full table: [token_num, max_topk]
  const int exp_idx = expert_ids_full[expert_slice_start + blockIdx.y];
  if (exp_idx > 255 || exp_idx < 0) return;


  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_K128* y = (const block_q8_K128*)(vy);

  float sum[mmq_y / (row_per_warp*nwarps)] = {0.0f};
  const int col_y_eff = token_offs;
  const block_q8_K128* yy = (const block_q8_K128*)(y + col_y_eff * blocks_per_col_y);
 
  __shared__ uint64_t shared_iq1s_grid[2048];

  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;

  const int vec2_elements = 2048 / 2;
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;

  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_gpu);
  ulonglong2* shared_iq1s_grid_vec2 = reinterpret_cast<ulonglong2*>(shared_iq1s_grid);
  #pragma unroll
  for (int i = 0; i < vec2_per_thread; i++) {
      int j = tid + i * block_size;
      if (j < vec2_elements) {
          shared_iq1s_grid_vec2[j] = __ldg(&iq1s_grid_vec2[j]);
      }
  }
  __syncthreads();

#pragma unroll
  for(int k = 0; k < mmq_y; k += row_per_warp*nwarps){
    float d;
    const int row_offs = (row + k) / 4;
    const int iqs = threadIdx.x % 4;
    const half * df = (const half *) ((const char *)x + (row_offs) * row_size * 4);
    d = __half2float(df[iqs]);
    const block_q_t* xx = (const block_q_t*)(df+4);
    #pragma unroll
    for (int i = 0; i < blocks_per_row_x; i += reduce_per_warp) {
      const int ibx = i + threadIdx.x / row_per_warp;
      const int iby = (ibx * qk/(128));
      const int iqa = ibx % 4;
      sum[k/(row_per_warp*nwarps)] += vec_dot(&xx[ibx], &yy[iby], iqs, iqa, d, shared_iq1s_grid);
    }
  }

#pragma unroll
  for (int i = 0; i < mmq_y; i += row_per_warp*nwarps) {
    const int row_dst = row_dst_0 + threadIdx.x % row_per_warp +  row_per_warp * threadIdx.y + i;
    if (row_dst >= nrows_dst) {
      continue;
    }
    #pragma unroll
    for (int mask = 16; mask >= row_per_warp; mask >>= 1){
      sum[i / (row_per_warp*nwarps)] += VLLM_SHFL_XOR_SYNC(sum[i / (row_per_warp*nwarps)], mask);
    }
    if(threadIdx.x / row_per_warp == 0){
      dst[col_dst_0 * nrows_dst + row_dst] = sum[i / (row_per_warp*nwarps)];
    }
  }
}

template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_r4_shared_cuda_t_16b vec_dot>
static __device__ __forceinline__ void moe_vec_q_shared_1_dyn_prefill_16b(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst,
    const int max_topk) {
  const int reduce_per_warp = qi;
  const int row_per_warp = WARP_SIZE_GGUF / reduce_per_warp;

  const int blocks_per_row_x = ncols_x / qk;
  // Q8_K128 每个block处理128
  const int blocks_per_col_y = nrows_y / 128;
  const int row_size = std::is_same<block_q_t, block_iq1_s_r4>::value ?  (2 + ncols_x * 1.5 / 8) :
                       std::is_same<block_q_t, block_iq1_m_r4>::value ?
                       (2 + ncols_x * 7 / 32) : 0;

  const int ncols_dst = ncols_y;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int row = row_dst_0 + threadIdx.x % row_per_warp + row_per_warp * threadIdx.y;

  const int idx = __ldg(idx_dev);
  const int top_k = __ldg(topk_dev);
  const int sorted_slice_start = __ldg(sorted_slice_start_dev);
  const int expert_slice_start = __ldg(expert_slice_start_dev);

  // Replicate host-side slice_start logic (must stay deterministic for capture).

  const int col_dst_0 = blockIdx.y * mmq_x;
  const int limit = num_tokens_post_padded_all[idx];
  if (blockIdx.y * mmq_x >= limit) return;
  // 这里默认mmq_x = 1
  const int token_ids = sorted_token_ids_all[sorted_slice_start + col_dst_0];
  const int token_offs = token_ids / top_k;

  // Compute expert id from full table: [token_num, max_topk]
  const int exp_idx = expert_ids_full[expert_slice_start + blockIdx.y];
  if (exp_idx > 255 || exp_idx < 0) return;


  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_K128* y = (const block_q8_K128*)(vy);

  float sum[mmq_y / (row_per_warp*nwarps)] = {0.0f};
  const int col_y_eff = token_offs;
  const block_q8_K128* yy = (const block_q8_K128*)(y + col_y_eff * blocks_per_col_y);
 /*
  __shared__ uint64_t shared_iq1s_grid[2048];

  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;

  const int vec2_elements = 2048 / 2;
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;

  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_gpu);
  ulonglong2* shared_iq1s_grid_vec2 = reinterpret_cast<ulonglong2*>(shared_iq1s_grid);
  #pragma unroll
  for (int i = 0; i < vec2_per_thread; i++) {
      int j = tid + i * block_size;
      if (j < vec2_elements) {
          shared_iq1s_grid_vec2[j] = __ldg(&iq1s_grid_vec2[j]);
      }
  }

  __syncthreads();
  */
  /*
  // 1. 类型改为 uint16_t，并强制 16 字节对齐，防止强转 ulonglong2 时报错
  __align__(16) __shared__ uint16_t shared_iq1s_grid[2048];

  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;

  // 2. 重新计算 128-bit 向量的加载次数
  // 总容量: 2048 个 uint16_t = 4096 字节
  // 每次读取: 1 个 ulonglong2 = 16 字节
  // 需要读取的次数: 4096 / 16 = 256 次
  const int vec2_elements = 256; 
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;

  // 3. 全局数组名请替换为你实际生成的 16-bit 数组名 (例如 iq1s_grid_16bit)
  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_16bit);
  ulonglong2* shared_iq1s_grid_vec2 = reinterpret_cast<ulonglong2*>(shared_iq1s_grid);
  
  #pragma unroll
  for (int i = 0; i < vec2_per_thread; i++) {
      int j = tid + i * block_size;
      if (j < vec2_elements) {
          shared_iq1s_grid_vec2[j] = __ldg(&iq1s_grid_vec2[j]);
      }
  }

  __syncthreads();
*/

  __shared__ uint32_t shared_grid0[2048];
  __shared__ uint32_t shared_grid1[2048];

  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;

  // 依然使用 128-bit 的 ulonglong2 向量化读取以打满 Global Memory 带宽
  const int vec2_elements = 2048 / 2;
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;
  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_gpu);

  #pragma unroll
  for (int i = 0; i < vec2_per_thread; i++) {
      int j = tid + i * block_size;
      if (j < vec2_elements) {
          // 从 Global Memory 读取 128-bit 数据 (包含两个 uint64_t)
          ulonglong2 vals = __ldg(&iq1s_grid_vec2[j]);

          // 原代码中读取 grid 时隐式转换为了 int (32-bit)，这里保持一致只取低 32 位
          uint32_t val0 = static_cast<uint32_t>(vals.x);
          uint32_t val1 = static_cast<uint32_t>(vals.y);

          // --- 修改点 2：在写入 Shared Memory 时提前完成解包和位运算 ---
          // 处理第一个元素 (索引 j * 2)
          shared_grid0[j * 2]     = val0 & 0x0F0F0F0F;
          shared_grid1[j * 2]     = (val0 >> 4) & 0x0F0F0F0F;

          // 处理第二个元素 (索引 j * 2 + 1)
          shared_grid0[j * 2 + 1] = val1 & 0x0F0F0F0F;
          shared_grid1[j * 2 + 1] = (val1 >> 4) & 0x0F0F0F0F;
      }
  }
  __syncthreads();
#pragma unroll
  for(int k = 0; k < mmq_y; k += row_per_warp*nwarps){
    float d;
    const int row_offs = (row + k) / 4;
    const int iqs = threadIdx.x % 4;
    const half * df = (const half *) ((const char *)x + (row_offs) * row_size * 4);
    d = __half2float(df[iqs]);
    const block_q_t* xx = (const block_q_t*)(df+4);
    #pragma unroll
    for (int i = 0; i < blocks_per_row_x; i += reduce_per_warp) {
      const int ibx = i + threadIdx.x / row_per_warp;
      const int iby = (ibx * qk/(128));
      const int iqa = ibx % 4;
      sum[k/(row_per_warp*nwarps)] += vec_dot_iq1_m_r4_q8_k128_shared_double(&xx[ibx], &yy[iby], iqs, iqa, d, shared_grid0, shared_grid1);
    }
  }

#pragma unroll
  for (int i = 0; i < mmq_y; i += row_per_warp*nwarps) {
    const int row_dst = row_dst_0 + threadIdx.x % row_per_warp +  row_per_warp * threadIdx.y + i;
    if (row_dst >= nrows_dst) {
      continue;
    }
    #pragma unroll
    for (int mask = 16; mask >= row_per_warp; mask >>= 1){
      sum[i / (row_per_warp*nwarps)] += VLLM_SHFL_XOR_SYNC(sum[i / (row_per_warp*nwarps)], mask);
    }
    if(threadIdx.x / row_per_warp == 0){
      dst[col_dst_0 * nrows_dst + row_dst] = sum[i / (row_per_warp*nwarps)];
    }
  }
}
template <typename scalar_t, bool need_check>
static __global__ void moe_vec_iq1_s_r4_dyn(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int max_topk) {
  const int mmq_x = MMQ_X_IQ1_R4;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  moe_vec_q_shared_1_dyn<scalar_t, 32, 2, 8, true, block_iq1_s_r4, mmq_x, mmq_y, nwarps,
      allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
      1, vec_dot_iq1_s_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
      idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
}

template <typename scalar_t, bool need_check>
static __global__ void moe_vec_iq1_s_r4_dyn_prefill(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int max_topk) {
  const int mmq_x = MMQ_X_IQ1_R4;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  moe_vec_q_shared_1_dyn_prefill<scalar_t, 32, 2, 8, true, block_iq1_s_r4, mmq_x, mmq_y, nwarps,
      allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
      1, vec_dot_iq1_s_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
      idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
}

template <typename scalar_t, bool need_check>
static __global__ void moe_vec_iq1_s_r4_dyn_w2(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ flex_topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int top_k,
    const int max_topk) {
  const int mmq_x = MMQ_X_IQ1_R4;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  moe_vec_q_shared_1_dyn_w2<scalar_t, 32, 2, 8, true, block_iq1_s_r4, mmq_x, mmq_y, nwarps,
      allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
      1, vec_dot_iq1_s_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
      idx_dev, flex_topk_dev, sorted_slice_start_dev,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
}

template <typename scalar_t, bool need_check>
static __global__ void moe_vec_iq1_s_r4_dyn_w2_prefill(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int top_k,
    const int max_topk) {
  const int mmq_x = MMQ_X_IQ1_R4;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  moe_vec_q_shared_1_dyn_w2_prefill<scalar_t, 32, 2, 8, true, block_iq1_s_r4, mmq_x, mmq_y, nwarps,
      allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
      1, vec_dot_iq1_s_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
      idx_dev, expert_slice_start_dev, sorted_slice_start_dev,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
}


template <typename scalar_t, bool need_check>
static __global__ void moe_vec_iq1_m_r4_dyn(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int max_topk) {
  const int mmq_x = MMQ_X_IQ1_R4;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  moe_vec_q_shared_1_dyn<scalar_t, 32, 2, 8, true, block_iq1_m_r4, mmq_x, mmq_y, nwarps,
      allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
      1, vec_dot_iq1_m_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
      idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
}

template <typename scalar_t, bool need_check>
static __global__ void moe_vec_iq1_m_r4_dyn_prefill(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int max_topk) {
  const int mmq_x = MMQ_X_IQ1_R4;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  moe_vec_q_shared_1_dyn_prefill<scalar_t, 32, 2, 8, true, block_iq1_m_r4, mmq_x, mmq_y, nwarps,
      allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
      1, vec_dot_iq1_m_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
      idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
}

template <typename scalar_t, bool need_check, int batch_size>
static __global__ void moe_vec_iq1_m_r4_dyn_prefill_batch(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int max_topk) {
  static_assert(batch_size == 1 || batch_size == 2 || batch_size == 4 || batch_size == 8,
                "moe prefill batch must be one of {1,2,4,8}");
  constexpr int qk = 32;
  constexpr int qi = 8;
  constexpr int reduce_per_warp = qi;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  const int row_per_warp = WARP_SIZE_GGUF / reduce_per_warp;
  constexpr int max_sum_tiles = 4;

  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / 128;
  const int row_size = 2 + ncols_x * 7 / 32;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int row = row_dst_0 + threadIdx.x % row_per_warp + row_per_warp * threadIdx.y;

  const int idx = __ldg(idx_dev);
  const int top_k = __ldg(topk_dev);
  const int sorted_slice_start = __ldg(sorted_slice_start_dev);
  const int expert_slice_start = __ldg(expert_slice_start_dev);
  const int limit = num_tokens_post_padded_all[idx];

  const int col_dst_base = blockIdx.y * batch_size;
  if (col_dst_base >= limit) return;

  const block_q8_K128* y = (const block_q8_K128*)(vy);
  const block_q8_K128* yy[batch_size];
  bool valid[batch_size];
  bool done[batch_size];
  bool active[batch_size];
  int exp[batch_size];
  float sum[batch_size][max_sum_tiles];

  #pragma unroll
  for (int c = 0; c < batch_size; ++c) {
    yy[c] = y;
    valid[c] = false;
    done[c] = false;
    active[c] = false;
    exp[c] = -1;
    #pragma unroll
    for (int t = 0; t < max_sum_tiles; ++t) {
      sum[c][t] = 0.0f;
    }
  }

  #pragma unroll
  for (int c = 0; c < batch_size; ++c) {
    const int col = col_dst_base + c;
    if (col >= limit) continue;
    const int token_id = sorted_token_ids_all[sorted_slice_start + col];
    const int token_offs = token_id / top_k;
    exp[c] = expert_ids_full[expert_slice_start + col];
    valid[c] = exp[c] >= 0 && exp[c] < 256;
    if (valid[c]) yy[c] = y + token_offs * blocks_per_col_y;
  }

  __shared__ uint64_t shared_iq1s_grid[2048];
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;
  const int vec2_elements = 2048 / 2;
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;
  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_gpu);
  ulonglong2* shared_iq1s_grid_vec2 = reinterpret_cast<ulonglong2*>(shared_iq1s_grid);
  #pragma unroll
  for (int i = 0; i < vec2_per_thread; i++) {
      int j = tid + i * block_size;
      if (j < vec2_elements) {
          shared_iq1s_grid_vec2[j] = __ldg(&iq1s_grid_vec2[j]);
      }
  }
  __syncthreads();

  #pragma unroll
  for (int k = 0; k < mmq_y; k += row_per_warp*nwarps) {
    const int row_offs = (row + k) / 4;
    const int iqs = threadIdx.x % 4;
    const int kt = k / (row_per_warp * nwarps);

    #pragma unroll
    for (int c = 0; c < batch_size; ++c) {
      done[c] = false;
    }

    #pragma unroll
    for (int leader = 0; leader < batch_size; ++leader) {
      if (!valid[leader] || done[leader]) continue;
      const int exp_idx = exp[leader];

      #pragma unroll
      for (int c = 0; c < batch_size; ++c) {
        active[c] = valid[c] && !done[c] && (exp[c] == exp_idx);
        if (active[c]) done[c] = true;
      }

      const block_iq1_m_r4* x = (const block_iq1_m_r4*)((const char*)vx + exp_idx * exp_stride);
      const half * df = (const half *) ((const char *)x + (row_offs) * row_size * 4);
      const float d = __half2float(df[iqs]);
      const block_iq1_m_r4* xx = (const block_iq1_m_r4*)(df + 4);

      for (int i = 0; i < blocks_per_row_x; i += reduce_per_warp) {
        const int ibx = i + threadIdx.x / row_per_warp;
        const int iby = (ibx * qk / 128);
        const int iqa = ibx % 4;
        const block_q8_K128* bq8_ptrs[batch_size];
        float out[batch_size];
        #pragma unroll
        for (int c = 0; c < batch_size; ++c) {
          bq8_ptrs[c] = &yy[c][iby];
          out[c] = sum[c][kt];
        }

        vec_dot_iq1_m_r4_q8_k128_shared_batch<batch_size>(
            &xx[ibx], bq8_ptrs, active, iqs, iqa, d, shared_iq1s_grid, out);

        #pragma unroll
        for (int c = 0; c < batch_size; ++c) {
          sum[c][kt] = out[c];
        }
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < mmq_y; i += row_per_warp*nwarps) {
    const int row_dst = row_dst_0 + threadIdx.x % row_per_warp + row_per_warp * threadIdx.y + i;
    if (row_dst >= nrows_dst) {
      continue;
    }

    #pragma unroll
    for (int mask = 16; mask >= row_per_warp; mask >>= 1) {
      #pragma unroll
      for (int c = 0; c < batch_size; ++c) {
        sum[c][i / (row_per_warp*nwarps)] += VLLM_SHFL_XOR_SYNC(sum[c][i / (row_per_warp*nwarps)], mask);
      }
    }
    if (threadIdx.x / row_per_warp == 0) {
      #pragma unroll
      for (int c = 0; c < batch_size; ++c) {
        if (valid[c]) {
          dst[(col_dst_base + c) * nrows_dst + row_dst] = sum[c][i / (row_per_warp*nwarps)];
        }
      }
    }
  }
}

template <typename scalar_t, bool need_check>
static __global__ void moe_vec_iq1_m_r4_dyn_w2(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ flex_topk_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int top_k,
    const int max_topk) {
  const int mmq_x = MMQ_X_IQ1_R4;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  moe_vec_q_shared_1_dyn_w2<scalar_t, 32, 2, 8, true, block_iq1_m_r4, mmq_x, mmq_y, nwarps,
      allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
      1, vec_dot_iq1_m_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
      idx_dev, flex_topk_dev, sorted_slice_start_dev,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
}

template <typename scalar_t, bool need_check, int batch_size>
static __global__ void moe_vec_iq1_m_r4_dyn_w2_prefill_batch(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int top_k,
    const int max_topk) {
  static_assert(batch_size == 1 || batch_size == 2 || batch_size == 4 || batch_size == 8,
                "moe w2 prefill batch must be one of {1,2,4,8}");
  constexpr int qk = 32;
  constexpr int qi = 8;
  constexpr int reduce_per_warp = qi;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  const int row_per_warp = WARP_SIZE_GGUF / reduce_per_warp;
  constexpr int max_sum_tiles = 4;

  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / 128;
  const int row_size = 2 + ncols_x * 7 / 32;

  const int row_dst_0 = blockIdx.x * mmq_y;
  const int row = row_dst_0 + threadIdx.x % row_per_warp + row_per_warp * threadIdx.y;

  const int idx = __ldg(idx_dev);
  const int sorted_slice_start = __ldg(sorted_slice_start_dev);
  const int expert_slice_start = __ldg(expert_slice_start_dev);
  const int limit = num_tokens_post_padded_all[idx];

  const int col_dst_base = blockIdx.y * batch_size;
  if (col_dst_base >= limit) return;

  const block_q8_K128* y = (const block_q8_K128*)(vy);
  const block_q8_K128* yy[batch_size];
  bool valid[batch_size];
  bool done[batch_size];
  bool active[batch_size];
  int exp[batch_size];
  float sum[batch_size][max_sum_tiles];

  #pragma unroll
  for (int c = 0; c < batch_size; ++c) {
    yy[c] = y;
    valid[c] = false;
    done[c] = false;
    active[c] = false;
    exp[c] = -1;
    #pragma unroll
    for (int t = 0; t < max_sum_tiles; ++t) {
      sum[c][t] = 0.0f;
    }
  }

  #pragma unroll
  for (int c = 0; c < batch_size; ++c) {
    const int col = col_dst_base + c;
    if (col >= limit) continue;
    const int token_id = sorted_token_ids_all[sorted_slice_start + col];
    const int token_offs = token_id / top_k;
    exp[c] = expert_ids_full[expert_slice_start + col];
    valid[c] = exp[c] >= 0 && exp[c] < 256;
    if (valid[c]) yy[c] = y + token_offs * blocks_per_col_y;
  }

  __shared__ uint64_t shared_iq1s_grid[2048];
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;
  const int vec2_elements = 2048 / 2;
  const int vec2_per_thread = (vec2_elements + block_size - 1) / block_size;
  const ulonglong2* iq1s_grid_vec2 = reinterpret_cast<const ulonglong2*>(iq1s_grid_gpu);
  ulonglong2* shared_iq1s_grid_vec2 = reinterpret_cast<ulonglong2*>(shared_iq1s_grid);
  #pragma unroll
  for (int i = 0; i < vec2_per_thread; i++) {
      int j = tid + i * block_size;
      if (j < vec2_elements) {
          shared_iq1s_grid_vec2[j] = __ldg(&iq1s_grid_vec2[j]);
      }
  }
  __syncthreads();

  #pragma unroll
  for (int k = 0; k < mmq_y; k += row_per_warp*nwarps) {
    const int row_offs = (row + k) / 4;
    const int iqs = threadIdx.x % 4;
    const int kt = k / (row_per_warp * nwarps);

    #pragma unroll
    for (int c = 0; c < batch_size; ++c) {
      done[c] = false;
    }

    #pragma unroll
    for (int leader = 0; leader < batch_size; ++leader) {
      if (!valid[leader] || done[leader]) continue;
      const int exp_idx = exp[leader];

      #pragma unroll
      for (int c = 0; c < batch_size; ++c) {
        active[c] = valid[c] && !done[c] && (exp[c] == exp_idx);
        if (active[c]) done[c] = true;
      }

      const block_iq1_m_r4* x = (const block_iq1_m_r4*)((const char*)vx + exp_idx * exp_stride);
      const half * df = (const half *) ((const char *)x + (row_offs) * row_size * 4);
      const float d = __half2float(df[iqs]);
      const block_iq1_m_r4* xx = (const block_iq1_m_r4*)(df + 4);

      for (int i = 0; i < blocks_per_row_x; i += reduce_per_warp) {
        const int ibx = i + threadIdx.x / row_per_warp;
        const int iby = (ibx * qk / 128);
        const int iqa = ibx % 4;
        const block_q8_K128* bq8_ptrs[batch_size];
        float out[batch_size];
        #pragma unroll
        for (int c = 0; c < batch_size; ++c) {
          bq8_ptrs[c] = &yy[c][iby];
          out[c] = sum[c][kt];
        }

        vec_dot_iq1_m_r4_q8_k128_shared_batch<batch_size>(
            &xx[ibx], bq8_ptrs, active, iqs, iqa, d, shared_iq1s_grid, out);

        #pragma unroll
        for (int c = 0; c < batch_size; ++c) {
          sum[c][kt] = out[c];
        }
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < mmq_y; i += row_per_warp*nwarps) {
    const int row_dst = row_dst_0 + threadIdx.x % row_per_warp + row_per_warp * threadIdx.y + i;
    if (row_dst >= nrows_dst) {
      continue;
    }

    #pragma unroll
    for (int mask = 16; mask >= row_per_warp; mask >>= 1) {
      #pragma unroll
      for (int c = 0; c < batch_size; ++c) {
        sum[c][i / (row_per_warp*nwarps)] += VLLM_SHFL_XOR_SYNC(sum[c][i / (row_per_warp*nwarps)], mask);
      }
    }
    if (threadIdx.x / row_per_warp == 0) {
      #pragma unroll
      for (int c = 0; c < batch_size; ++c) {
        if (valid[c]) {
          dst[(col_dst_base + c) * nrows_dst + row_dst] = sum[c][i / (row_per_warp*nwarps)];
        }
      }
    }
  }
}

template <typename scalar_t, bool need_check>
static __global__ void moe_vec_iq1_m_r4_dyn_w2_prefill(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids_all,
    const int* __restrict__ expert_ids_full,
    const int* __restrict__ num_tokens_post_padded_all,
    const int* __restrict__ idx_dev,
    const int* __restrict__ expert_slice_start_dev,
    const int* __restrict__ sorted_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int top_k,
    const int max_topk) {
  const int mmq_x = MMQ_X_IQ1_R4;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  moe_vec_q_shared_1_dyn_w2_prefill<scalar_t, 32, 2, 8, true, block_iq1_m_r4, mmq_x, mmq_y, nwarps,
      allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
      1, vec_dot_iq1_m_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
      idx_dev, expert_slice_start_dev, sorted_slice_start_dev,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
}


template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_0, 2)
#endif
    moe_vec_iq1_r4(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MMQ_X_IQ1_R4;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  // moe_vec_q_shared_1 在并发少的情况下性能较好，并行度取8
  moe_vec_q_shared_1<scalar_t, 32, 2, 8, true, block_iq1_s_r4, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
        1, vec_dot_iq1_s_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);

  /*moe_vec_q_shared<scalar_t, 32, 2, 4, true, block_iq1_s_r4, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
        1, vec_dot_iq1_s_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
      */
}


template <typename scalar_t>
static void ggml_moe_vec_iq1_s_r4_q8_k128_cuda_dyn(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids_all,
    const int* expert_ids_full,
    const int* num_tokens_post_padded_all,
    const int* idx_dev,
    const int* topk_dev,
    const int* sorted_slice_start_dev,
    const int* expert_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int max_topk,
    const int tokens_post_padded_max,
    cudaStream_t stream) {
  int mmq_x = MMQ_X_IQ1_R4;
  int mmq_y = MMQ_Y_IQ1_R4;
  int nwarps = NWARPS_IQ1_R4;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded_max) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_s_r4_dyn<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
  } else {
    constexpr bool need_check = true;
    moe_vec_iq1_s_r4_dyn<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
  }
}

template <typename scalar_t>
static void ggml_moe_vec_iq1_s_r4_q8_k128_cuda_dyn_prefill(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids_all,
    const int* expert_ids_full,
    const int* num_tokens_post_padded_all,
    const int* idx_dev,
    const int* topk_dev,
    const int* sorted_slice_start_dev,
    const int* expert_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int max_topk,
    const int tokens_post_padded_max,
    cudaStream_t stream) {
  int mmq_x = MMQ_X_IQ1_R4;
  int mmq_y = MMQ_Y_IQ1_R4;
  int nwarps = NWARPS_IQ1_R4;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded_max) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_s_r4_dyn_prefill<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
  } else {
    constexpr bool need_check = true;
    moe_vec_iq1_s_r4_dyn_prefill<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
  }
}

template <typename scalar_t>
static void ggml_moe_vec_iq1_s_r4_q8_k128_cuda_dyn_w2(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids_all,
    const int* expert_ids_full,
    const int* num_tokens_post_padded_all,
    const int* idx_dev,
    const int* flex_topk_dev,
    const int* sorted_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int top_k,
    const int max_topk,
    const int tokens_post_padded_max,
    cudaStream_t stream) {
  int mmq_x = MMQ_X_IQ1_R4;
  int mmq_y = MMQ_Y_IQ1_R4;
  int nwarps = NWARPS_IQ1_R4;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded_max) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_s_r4_dyn_w2<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, flex_topk_dev, sorted_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
  } else {
    constexpr bool need_check = true;
    moe_vec_iq1_s_r4_dyn_w2<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, flex_topk_dev, sorted_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
  }
}

template <typename scalar_t>
static void ggml_moe_vec_iq1_s_r4_q8_k128_cuda_dyn_w2_prefill(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids_all,
    const int* expert_ids_full,
    const int* num_tokens_post_padded_all,
    const int* idx_dev,
    const int* expert_slice_start_dev,
    const int* sorted_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int top_k,
    const int max_topk,
    const int tokens_post_padded_max,
    cudaStream_t stream) {
  int mmq_x = MMQ_X_IQ1_R4;
  int mmq_y = MMQ_Y_IQ1_R4;
  int nwarps = NWARPS_IQ1_R4;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded_max) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_s_r4_dyn_w2_prefill<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, expert_slice_start_dev, sorted_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
  } else {
    constexpr bool need_check = true;
    moe_vec_iq1_s_r4_dyn_w2_prefill<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, expert_slice_start_dev, sorted_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
  }
}

template <typename scalar_t>
static void ggml_moe_vec_iq1_m_r4_q8_k128_cuda_dyn_w2_prefill(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids_all,
    const int* expert_ids_full,
    const int* num_tokens_post_padded_all,
    const int* idx_dev,
    const int* expert_slice_start_dev,
    const int* sorted_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int top_k,
    const int max_topk,
    const int tokens_post_padded_max,
    cudaStream_t stream) {
  int mmq_y = MMQ_Y_IQ1_R4;
  int nwarps = NWARPS_IQ1_R4;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

#if MMQ_X_IQ1_M_R4_PREFILL_BATCH == 1
  const int block_num_y = (tokens_post_padded_max + MMQ_X_IQ1_R4 - 1) / MMQ_X_IQ1_R4;
  const dim3 block_nums(block_num_x, block_num_y, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_m_r4_dyn_w2_prefill<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, expert_slice_start_dev, sorted_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
  } else {
    constexpr bool need_check = true;
    moe_vec_iq1_m_r4_dyn_w2_prefill<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, expert_slice_start_dev, sorted_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
  }
#else
  const int block_num_y = (tokens_post_padded_max + MMQ_X_IQ1_M_R4_PREFILL_BATCH - 1) / MMQ_X_IQ1_M_R4_PREFILL_BATCH;
  const dim3 block_nums(block_num_x, block_num_y, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_m_r4_dyn_w2_prefill_batch<scalar_t, need_check, MMQ_X_IQ1_M_R4_PREFILL_BATCH><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, expert_slice_start_dev, sorted_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
  } else {
    constexpr bool need_check = true;
    moe_vec_iq1_m_r4_dyn_w2_prefill_batch<scalar_t, need_check, MMQ_X_IQ1_M_R4_PREFILL_BATCH><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, expert_slice_start_dev, sorted_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
  }
#endif
}

template <typename scalar_t>
static void ggml_moe_vec_iq1_m_r4_q8_k128_cuda_dyn_w2(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids_all,
    const int* expert_ids_full,
    const int* num_tokens_post_padded_all,
    const int* idx_dev,
    const int* flex_topk_dev,
    const int* sorted_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int top_k,
    const int max_topk,
    const int tokens_post_padded_max,
    cudaStream_t stream) {
  int mmq_x = MMQ_X_IQ1_R4;
  int mmq_y = MMQ_Y_IQ1_R4;
  int nwarps = NWARPS_IQ1_R4;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded_max) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_m_r4_dyn_w2<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, flex_topk_dev, sorted_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
  } else {
    constexpr bool need_check = true;
    moe_vec_iq1_m_r4_dyn_w2<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, flex_topk_dev, sorted_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k, max_topk);
  }
}

template <typename scalar_t>
static void ggml_moe_vec_iq1_m_r4_q8_k128_cuda_dyn(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids_all,
    const int* expert_ids_full,
    const int* num_tokens_post_padded_all,
    const int* idx_dev,
    const int* topk_dev,
    const int* sorted_slice_start_dev,
    const int* expert_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int max_topk,
    const int tokens_post_padded_max,
    cudaStream_t stream) {
  int mmq_x = MMQ_X_IQ1_R4;
  int mmq_y = MMQ_Y_IQ1_R4;
  int nwarps = NWARPS_IQ1_R4;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded_max) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_m_r4_dyn<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
  } else {
    constexpr bool need_check = true;
    moe_vec_iq1_m_r4_dyn<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
  }
}

template <typename scalar_t>
static void ggml_moe_vec_iq1_m_r4_q8_k128_cuda_dyn_prefill(
    const void* inp, const void* w, scalar_t* dst,
    const int* sorted_token_ids_all,
    const int* expert_ids_full,
    const int* num_tokens_post_padded_all,
    const int* idx_dev,
    const int* topk_dev,
    const int* sorted_slice_start_dev,
    const int* expert_slice_start_dev,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    const int max_topk,
    const int tokens_post_padded_max,
    cudaStream_t stream) {
  int mmq_y = MMQ_Y_IQ1_R4;
  int nwarps = NWARPS_IQ1_R4;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

#if MMQ_X_IQ1_M_R4_PREFILL_BATCH == 1
  const int block_num_y = (tokens_post_padded_max + MMQ_X_IQ1_R4 - 1) / MMQ_X_IQ1_R4;
  const dim3 block_nums(block_num_x, block_num_y, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_m_r4_dyn_prefill<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
  } else {
    constexpr bool need_check = true;
    moe_vec_iq1_m_r4_dyn_prefill<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
  }
#else
  const int block_num_y = (tokens_post_padded_max + MMQ_X_IQ1_M_R4_PREFILL_BATCH - 1) / MMQ_X_IQ1_M_R4_PREFILL_BATCH;
  const dim3 block_nums(block_num_x, block_num_y, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_m_r4_dyn_prefill_batch<scalar_t, need_check, MMQ_X_IQ1_M_R4_PREFILL_BATCH><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
  } else {
    constexpr bool need_check = true;
    moe_vec_iq1_m_r4_dyn_prefill_batch<scalar_t, need_check, MMQ_X_IQ1_M_R4_PREFILL_BATCH><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst,
        sorted_token_ids_all, expert_ids_full, num_tokens_post_padded_all,
        idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, max_topk);
  }
#endif
  }

template <typename scalar_t>
static void ggml_moe_vec_iq1_s_r4_q8_k128_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  int mmq_x = MMQ_X_IQ1_R4;
  int mmq_y = MMQ_Y_IQ1_R4;
  int nwarps = NWARPS_IQ1_R4;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (ncols_y) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);
// 可能会出现两种情况，一种是单向量乘多专家，一种是多向量乘多专家，但都是一对一的关系
  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_r4<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    // 边界检查
    constexpr bool need_check = true;
    moe_vec_iq1_r4<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_0, 2)
#endif
    moe_vec_iq1_m_r4(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MMQ_X_IQ1_R4;
  const int mmq_y = MMQ_Y_IQ1_R4;
  const int nwarps = NWARPS_IQ1_R4;
  // moe_vec_q_shared_1 在并发少的情况下性能较好，并行度取8
  moe_vec_q_shared_1<scalar_t, 32, 2, 8, true, block_iq1_m_r4, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
        1, vec_dot_iq1_m_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);

  /*moe_vec_q_shared<scalar_t, 32, 2, 4, true, block_iq1_s_r4, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
        1, vec_dot_iq1_s_r4_q8_k128_shared>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
      */
}

template <typename scalar_t>
static void ggml_moe_vec_iq1_m_r4_q8_k128_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  int mmq_x = MMQ_X_IQ1_R4;
  int mmq_y = MMQ_Y_IQ1_R4;
  int nwarps = NWARPS_IQ1_R4;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (ncols_y) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);
// 可能会出现两种情况，一种是单向量乘多专家，一种是多向量乘多专家，但都是一对一的关系
  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_vec_iq1_m_r4<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    // 边界检查
    constexpr bool need_check = true;
    moe_vec_iq1_m_r4<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}