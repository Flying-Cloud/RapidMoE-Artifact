/**
 * @Description  :
 * @Author       : Azure-Tang
 * @Date         : 2024-07-22 09:27:55
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-12 03:48:46
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
**/
#pragma once

#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor dequantize_q8_0(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype);
torch::Tensor dequantize_q6_k(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype);
torch::Tensor dequantize_q5_k(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype);
torch::Tensor dequantize_q4_k(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype);
torch::Tensor dequantize_q3_k(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype);
torch::Tensor dequantize_q2_k(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype);
torch::Tensor dequantize_iq4_xs(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const torch::Device device, const torch::Dtype target_dtype);
torch::Tensor dequantize_iq1_s_r4(const int8_t* data, const int num_bytes, const int blk_size, const int ele_per_blk, const int num_rows, const int n_per_row, const int row_size, const torch::Device device, const torch::Dtype target_dtype);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                              int64_t row);

torch::Tensor ggml_mul_mat_vec_q8_k128(torch::Tensor W, torch::Tensor X,
                                      int64_t type, int64_t row);

torch::Tensor ggml_moe_vec_q8_k128(torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded, int64_t type, int64_t row, int64_t top_k, int64_t tokens);

torch::Tensor moe_gemm(torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids_all, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded_all, int64_t type, int64_t row, torch::Tensor idx_dev, torch::Tensor topk_dev, torch::Tensor sorted_slice_start_dev, torch::Tensor expert_slice_start_dev, torch::Tensor expert_slice_end_dev, int64_t max_topk, int64_t max_tokens);

torch::Tensor moe_gemm_w2(torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids_all, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded_all, int64_t type, int64_t row, int64_t top_k, torch::Tensor idx_dev, torch::Tensor topk_dev, torch::Tensor sorted_slice_start_dev, int64_t max_topk, int64_t max_tokens);

torch::Tensor moe_gemm_prefill(torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids_all, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded_all, int64_t type, int64_t row,  torch::Tensor idx_dev, torch::Tensor topk_dev, torch::Tensor sorted_slice_start_dev, torch::Tensor expert_slice_start_dev, int64_t max_topk, int64_t max_tokens);

torch::Tensor moe_gemm_prefill_tensor(torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids_all, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded_all, int64_t type, int64_t row,  torch::Tensor idx_dev, torch::Tensor topk_dev, torch::Tensor sorted_slice_start_dev, torch::Tensor expert_slice_start_dev, int64_t max_topk, int64_t max_tokens);

torch::Tensor moe_gemm_w2_prefill(torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids_all, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded_all, int64_t type, int64_t row, int64_t top_k, torch::Tensor idx_dev, torch::Tensor expert_slice_start_dev, torch::Tensor sorted_slice_start_dev, int64_t max_topk, int64_t max_tokens);

torch::Tensor ggml_moe_a8(torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded, int64_t type, int64_t row, int64_t top_k, int64_t tokens);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void mul_and_silu(torch::Tensor& out, torch::Tensor& input);

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void moe_weight_sum(torch::Tensor& input, torch::Tensor & weight, torch::Tensor& output);

void moe_weight_sum_v1(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& output, torch::Tensor& flex_topk_dev, int64_t max_topk, int64_t num_tokens);

void dynamic_add(torch::Tensor& input1, torch::Tensor& input2, torch::Tensor& flex_topk_dev);

void dynamic_threshold(torch::Tensor& topk_weight,
                       torch::Tensor& alpha,
                       torch::Tensor& threshold,
                       torch::Tensor& flex_topk_dev,
                       torch::Tensor& flex_idx_dev,
                       torch::Tensor& bsz_tensor);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);
void moe_align_block_size_v1(torch::Tensor topk_ids_raw, int64_t topk_start, int64_t topk_end, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);


int64_t ggml_moe_get_block_size(int64_t type);
