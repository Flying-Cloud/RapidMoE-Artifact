/**
 * @Description  :
 * @Author       : Azure-Tang, Boxin Zhang
 * @Date         : 2024-07-25 13:38:30
 * @Version      : 0.2.2
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
**/

#include "custom_gguf/ops.h"
#ifdef KTRANSFORMERS_USE_CUDA
#include "gptq_marlin/ops.h"
#endif
// Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
// namespace py = pybind11;

PYBIND11_MODULE(KTransformersOps, m) {

    m.def("dequantize_q8_0", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q8_0((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q8_0 data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_q6_k", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q6_k((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q6_k data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_q5_k", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q5_k((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q5_k data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_q4_k", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q4_k((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q4_k data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_q3_k", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q3_k((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q3_k data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_q2_k", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_q2_k((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize q2_k data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));

    m.def("dequantize_iq4_xs", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_iq4_xs((int8_t*)data, num_bytes, blk_size, ele_per_blk, device, dtype);
        }, "Function to dequantize iq4_xs data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("device"), py::arg("target_dtype"));
    // wwx: add iq1_s_r4
    m.def("dequantize_iq1_s_r4", [](const intptr_t data, int num_bytes, int blk_size, const int ele_per_blk, const int n_per_row, const int num_rows, const int row_size, torch::Device device, py::object target_dtype) {
        torch::Dtype dtype = torch::python::detail::py_object_to_dtype(target_dtype);
        return dequantize_iq1_s_r4((int8_t*)data, num_bytes, blk_size, ele_per_blk, n_per_row, num_rows, row_size, device, dtype);
        }, "Function to dequantize iq1_s_r4 data.",
        py::arg("data"), py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"), py::arg("n_per_row"), py::arg("num_rows"), py::arg("row_size"), py::arg("device"), py::arg("target_dtype"));
    // mmvq kernel for GGML.
    m.def("ggml_mul_mat_vec_a8", [](torch::Tensor W, torch::Tensor X, int type, int row) {
        return ggml_mul_mat_vec_a8(W, X, type, row);
    }, "Function to perform GGML mmvq kernel.",
        py::arg("W"), py::arg("X"), py::arg("type"), py::arg("row"));
    
    m.def("ggml_mul_mat_a8", [](torch::Tensor W, torch::Tensor X, int type, int row) {
        return ggml_mul_mat_a8(W, X, type, row);
    }, "Function to perform GGML mmq kernel.",
        py::arg("W"), py::arg("X"), py::arg("type"), py::arg("row"));

    m.def("ggml_mul_mat_vec_q8_k128", [](torch::Tensor W, torch::Tensor X, int type, int row) {
        return ggml_mul_mat_vec_q8_k128(W, X, type, row);
    }, "Function to perform GGML mmvq kernel.",
        py::arg("W"), py::arg("X"), py::arg("type"), py::arg("row"));
    
    m.def("ggml_moe_vec_q8_k128", [](torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded, int64_t type, int64_t row, int64_t top_k, int64_t tokens) {
        return ggml_moe_vec_q8_k128(X, W, sorted_token_ids, expert_ids, num_tokens_post_padded, type, row, top_k, tokens);
    }, "Function to perform GGML moe kernel for one token vecdot.",
        py::arg("X"), py::arg("W"), py::arg("sorted_token_ids"), py::arg("expert_ids"), py::arg("num_tokens_post_padded"), py::arg("type"), py::arg("row"), py::arg("top_k"), py::arg("tokens"));

    m.def("moe_gemm", [](torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids_all, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded_all, int64_t type, int64_t row, torch::Tensor idx_dev, torch::Tensor topk_dev, torch::Tensor sorted_slice_start_dev, torch::Tensor expert_slice_start_dev, torch::Tensor expert_slice_end_dev, int64_t max_topk, int64_t max_tokens) {
        return moe_gemm(X, W, sorted_token_ids_all, expert_ids, num_tokens_post_padded_all, type, row, idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev, expert_slice_end_dev, max_topk, max_tokens);
    }, "Function to perform moe gemm kernel.",
        py::arg("X"), py::arg("W"), py::arg("sorted_token_ids_all"), py::arg("expert_ids"), py::arg("num_tokens_post_padded_all"), py::arg("type"), py::arg("row"), py::arg("idx_dev"), py::arg("topk_dev"), py::arg("sorted_slice_start_dev"), py::arg("expert_slice_start_dev"), py::arg("expert_slice_end_dev"), py::arg("max_topk"), py::arg("max_tokens"));
    m.def("moe_gemm_w2", [](torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids_all, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded_all, int64_t type, int64_t row, int64_t top_k, torch::Tensor idx_dev, torch::Tensor topk_dev, torch::Tensor sorted_slice_start_dev, int64_t max_topk, int64_t max_tokens) {
        return moe_gemm_w2(X, W, sorted_token_ids_all, expert_ids, num_tokens_post_padded_all, type, row, top_k, idx_dev, topk_dev, sorted_slice_start_dev, max_topk, max_tokens);
    }, "Function to perform moe gemm down kernel.",
        py::arg("X"), py::arg("W"), py::arg("sorted_token_ids_all"), py::arg("expert_ids"), py::arg("num_tokens_post_padded_all"), py::arg("type"), py::arg("row"), py::arg("top_k"), py::arg("idx_dev"), py::arg("topk_dev"), py::arg("sorted_slice_start_dev"), py::arg("max_topk"), py::arg("max_tokens"));
    m.def("moe_gemm_prefill", [](torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids_all, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded_all, int64_t type, int64_t row,  torch::Tensor idx_dev, torch::Tensor topk_dev, torch::Tensor sorted_slice_start_dev, torch::Tensor expert_slice_start_dev, int64_t max_topk, int64_t max_tokens) { 
        return moe_gemm_prefill(X, W, sorted_token_ids_all, expert_ids, num_tokens_post_padded_all, type, row, idx_dev, topk_dev,sorted_slice_start_dev, expert_slice_start_dev, max_topk, max_tokens);
    }, "Function to perform moe gemm prefill kernel.",
        py::arg("X"), py::arg("W"), py::arg("sorted_token_ids_all"), py::arg("expert_ids"), py::arg("num_tokens_post_padded_all"), py::arg("type"), py::arg("row"), py::arg("idx_dev"), py::arg("topk_dev"), py::arg("sorted_slice_start_dev"), py::arg("expert_slice_start_dev"), py::arg("max_topk"), py::arg("max_tokens"));
    m.def("moe_gemm_prefill_tensor", [](torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids_all, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded_all, int64_t type, int64_t row,  torch::Tensor idx_dev, torch::Tensor topk_dev, torch::Tensor sorted_slice_start_dev, torch::Tensor expert_slice_start_dev, int64_t max_topk, int64_t max_tokens) {
        return moe_gemm_prefill_tensor(X, W, sorted_token_ids_all, expert_ids, num_tokens_post_padded_all, type, row, idx_dev, topk_dev, sorted_slice_start_dev, expert_slice_start_dev, max_topk, max_tokens);
    }, "Function to perform moe gemm prefill tensor-core path.",
        py::arg("X"), py::arg("W"), py::arg("sorted_token_ids_all"), py::arg("expert_ids"), py::arg("num_tokens_post_padded_all"), py::arg("type"), py::arg("row"), py::arg("idx_dev"), py::arg("topk_dev"), py::arg("sorted_slice_start_dev"), py::arg("expert_slice_start_dev"), py::arg("max_topk"), py::arg("max_tokens"));
    m.def("moe_gemm_w2_prefill", [](torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids_all, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded_all, int64_t type, int64_t row, int64_t top_k, torch::Tensor idx_dev, torch::Tensor expert_slice_start_dev, torch::Tensor sorted_slice_start_dev, int64_t max_topk, int64_t max_tokens) {
        return moe_gemm_w2_prefill(X, W, sorted_token_ids_all, expert_ids, num_tokens_post_padded_all, type, row, top_k, idx_dev, expert_slice_start_dev, sorted_slice_start_dev, max_topk, max_tokens);
    }, "Function to perform moe gemm down prefill kernel.",
        py::arg("X"), py::arg("W"), py::arg("sorted_token_ids_all"), py::arg("expert_ids"), py::arg("num_tokens_post_padded_all"), py::arg("type"), py::arg("row"), py::arg("top_k"), py::arg("idx_dev"), py::arg("expert_slice_start_dev"), py::arg("sorted_slice_start_dev"), py::arg("max_topk"), py::arg("max_tokens"));
    m.def("ggml_moe_a8", [](torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded, int64_t type, int64_t row, int64_t top_k, int64_t tokens) {
        return ggml_moe_a8(X, W, sorted_token_ids, expert_ids, num_tokens_post_padded, type, row, top_k, tokens);
    }, "Function to perform GGML moe kernel.",
        py::arg("X"), py::arg("W"), py::arg("sorted_token_ids"), py::arg("expert_ids"), py::arg("num_tokens_post_padded"), py::arg("type"), py::arg("row"), py::arg("top_k"), py::arg("tokens"));

    m.def("silu_and_mul", [](torch::Tensor out, torch::Tensor input) {
        silu_and_mul(out, input);
        return out;
    }, "Function to perform silu and mul kernel.",
        py::arg("out"), py::arg("input"));

    m.def("mul_and_silu", [](torch::Tensor out, torch::Tensor input) {
        mul_and_silu(out, input);
        return out;
    }, "Function to perform mul and silu kernel.",
        py::arg("out"), py::arg("input"));
    
    m.def("moe_sum", [](torch::Tensor input, torch::Tensor output) {
        moe_sum(input, output);
        return output;
    }, "Function to perform moe sum kernel.",
        py::arg("input"), py::arg("output"));

    m.def("moe_weight_sum", [](torch::Tensor input, torch::Tensor weight, torch::Tensor output) {
        moe_weight_sum(input, weight, output);
        return output;
    }, "Function to perform moe weight sum kernel.",
        py::arg("input"), py::arg("weight"), py::arg("output"));
    m.def("moe_weight_sum_v1", [](torch::Tensor input, torch::Tensor weight, torch::Tensor output, torch::Tensor flex_topk_dev, int64_t max_topk, int64_t num_tokens) {
        moe_weight_sum_v1(input, weight, output, flex_topk_dev, max_topk, num_tokens);
        return output;
    }, "Function to perform moe weight sum kernel.",
        py::arg("input"), py::arg("weight"), py::arg("output"), py::arg("flex_topk"), py::arg("max_topk"), py::arg("num_tokens"));
    
    m.def("dynamic_add", [](torch::Tensor input1, torch::Tensor input2, torch::Tensor flex_topk_dev) {
        dynamic_add(input1, input2, flex_topk_dev);
        return input1;
    }, "Function to perform dynamic add kernel.",
        py::arg("input1"), py::arg("input2"), py::arg("flex_topk_dev"));

    m.def("dynamic_threshold", [](torch::Tensor topk_weight, torch::Tensor alpha, torch::Tensor threshold, torch::Tensor flex_topk_dev, torch::Tensor flex_idx_dev, torch::Tensor bsz_tensor) {
        dynamic_threshold(topk_weight, alpha, threshold, flex_topk_dev, flex_idx_dev, bsz_tensor);
        return py::make_tuple(flex_topk_dev, flex_idx_dev);
    }, "Function to dynamically select topk by threshold.",
        py::arg("topk_weight"), py::arg("alpha"), py::arg("threshold"), py::arg("flex_topk_dev"), py::arg("flex_idx_dev"), py::arg("bsz_tensor"));

    m.def("moe_align_block_size", [](torch::Tensor topk_ids, int64_t num_experts, int64_t block_size, torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad) {
        moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, num_tokens_post_pad);
        return sorted_token_ids, experts_ids, num_tokens_post_pad;
    }, "Function to perform moe align block size kernel.",
        py::arg("topk_ids"), py::arg("num_experts"), py::arg("block_size"), py::arg("sorted_token_ids"), py::arg("experts_ids"), py::arg("num_tokens_post_pad"));

    m.def("moe_align_block_size_v1", [](torch::Tensor topk_ids_raw, int64_t topk_start, int64_t topk_end, int64_t num_experts, int64_t block_size, torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad) {
        moe_align_block_size_v1(topk_ids_raw, topk_start, topk_end, num_experts, block_size, sorted_token_ids, experts_ids, num_tokens_post_pad);
        return sorted_token_ids, experts_ids, num_tokens_post_pad;
    }, "Function to perform moe align block size kernel v1.",
        py::arg("topk_ids_raw"), py::arg("topk_start"), py::arg("topk_end"), py::arg("num_experts"), py::arg("block_size"), py::arg("sorted_token_ids"), py::arg("experts_ids"), py::arg("num_tokens_post_pad"));


    m.def("ggml_moe_get_block_size", [](int64_t type) {
        return ggml_moe_get_block_size(type);
    }, "Function to get moe block size.",
        py::arg("type"));

#ifdef KTRANSFORMERS_USE_CUDA
    m.def("gptq_marlin_gemm", &gptq_marlin_gemm, "Function to perform GEMM using Marlin quantization.",
        py::arg("a"), py::arg("b_q_weight"), py::arg("b_scales"), py::arg("g_idx"),
        py::arg("perm"), py::arg("workspace"), py::arg("num_bits"), py::arg("size_m"),
        py::arg("size_n"), py::arg("size_k"), py::arg("is_k_full"));
#endif
}