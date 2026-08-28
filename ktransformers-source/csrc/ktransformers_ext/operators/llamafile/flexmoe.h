/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_FLEXMOE_H
#define CPUINFER_OPERATOR_FLEXMOE_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>

#include "../../cpu_backend/backend.h"
#include "../../cpu_backend/shared_mem_buffer.h"
#include "conversion.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"
#include "llamafile/iqk_mul_mat.h"

struct FlexMOEConfig {
    int expert_num;
    int routed_expert_num;
    int flex_topk;
    int hidden_size;
    int intermediate_size;
    int stride;
    int group_min_len;
    int group_max_len;
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    ggml_type gate_type;
    ggml_type gate_ori_type;
    ggml_type gate_res_type;
    ggml_type up_type;
    ggml_type up_ori_type;
    ggml_type up_res_type;
    ggml_type down_type;
    ggml_type down_ori_type;
    ggml_type down_res_type;
    ggml_type hidden_type;


    FlexMOEConfig() {}

    FlexMOEConfig(int expert_num, int routed_expert_num, int flex_topk,int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, void* gate_proj, void* up_proj, void* down_proj, 
        ggml_type gate_type, ggml_type gate_ori_type, ggml_type gate_res_type, 
        ggml_type up_type,   ggml_type up_ori_type,   ggml_type up_res_type,
        ggml_type down_type, ggml_type down_ori_type, ggml_type down_res_type,
        ggml_type hidden_type)
        : expert_num(expert_num), routed_expert_num(routed_expert_num), flex_topk(flex_topk), hidden_size(hidden_size), intermediate_size(intermediate_size), stride(stride), group_min_len(group_min_len), group_max_len(group_max_len), gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), 
        gate_type(gate_type), gate_ori_type(gate_ori_type), gate_res_type(gate_res_type),
        up_type(up_type), up_ori_type(up_ori_type), up_res_type(up_res_type),
        down_type(down_type), down_ori_type(down_ori_type), down_res_type(down_res_type),
        hidden_type(hidden_type) {}
};

class FlexMOE {
   public:
    FlexMOE(FlexMOEConfig);
    ~FlexMOE();
    void warm_up(Backend* backend);
    void forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend);
    void forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend);
    void forward_many_v1(int qlen, int bsz, int k, int max_k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend);
    void forward_many_ftop(int qlen, int bsz, int k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, Backend* backend);
    void forward_many_ftop_v1(int qlen, int bsz, int k, int max_k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, Backend* backend);
    void forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, int* batch_size_tensor, Backend* backend);
    void forward_flex(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, Backend* backend);
    void forward_flex_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, int* batch_size_tensor, Backend* backend);
    void forward_flex_many_v1(int qlen, const uint64_t* k, int max_k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, int* batch_size_tensor, Backend* backend);
    void forward_many_flex(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, int* batch_size_tensor, Backend* backend);
    void forward_many_flex_v1(int qlen, const uint64_t* k, int max_k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, int* batch_size_tensor, Backend* backend);
    private:
    FlexMOEConfig config_;
    void* gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

    #ifdef USE_NUMA
    std::vector<void*> gate_proj_numa_;  // [numa_num, expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    std::vector<void*> up_proj_numa_;    // [numa_num, expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    std::vector<void*> down_proj_numa_;  // [numa_num, expert_num * hidden_size * intermediate_size ( /32 if quantized)]
    #endif

    float* s_input_fp32_;                      // [hidden_size]
    uint8_t* s_gate_input_;                    // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* s_up_input_;                      // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    std::vector<float*> s_gate_output_;        // [routed_expert_num, intermediate_size]
    std::vector<float*> s_up_output_;          // [routed_expert_num, intermediate_size]
    std::vector<float*> s_intermediate_fp32_;  // [routed_expert_num, intermediate_size]
    std::vector<uint8_t*> s_down_input_;       // [routed_expert_num, intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    std::vector<float*> s_down_output_;        // [routed_expert_num, hidden_size]
    float* s_output_fp32_;                     // [hidden_size]

    std::vector<float*> s_local_gate_gpu_input_;    // [flex_topk, intermediate_size]
    std::vector<float*> s_local_up_gpu_input_;      // [flex_topk, intermediate_size]

    std::vector<float*> m_input_fp32_;    // [group_max_len, hidden_size]
    std::vector<uint8_t*> m_gate_input_;  // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    std::vector<uint8_t*> m_up_input_;    // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    uint8_t* m_local_gate_input_;         // [routed_expert_num * group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* m_local_up_input_;           // [routed_expert_num * group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    float* m_local_gate_output_;          // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_up_output_;            // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_intermediate_fp32_;    // [routed_expert_num * group_max_len * intermediate_size]
    uint8_t* m_local_down_input_;         // [routed_expert_num * group_max_len * intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    float* m_local_down_output_;          // [routed_expert_num * group_max_len * hidden_size]
    std::vector<float*> m_output_fp32_;   // [group_max_len, hidden_size]

    std::vector<std::vector<int>> m_local_pos_;          // [group_max_len, routed_expert_num]
    std::vector<int> m_local_num_;                       // [expert_num]
    std::vector<int> m_local_offset_;                    // [expert_num]    
    std::vector<uint8_t*> m_local_gate_input_ptr_;       // [expert_num]
    std::vector<uint8_t*> m_local_up_input_ptr_;         // [expert_num]
    std::vector<float*> m_local_gate_output_ptr_;        // [expert_num]
    std::vector<float*> m_local_up_output_ptr_;          // [expert_num]
    std::vector<float*> m_local_intermediate_fp32_ptr_;  // [expert_num]
    std::vector<uint8_t*> m_local_down_input_ptr_;       // [expert_num]
    std::vector<float*> m_local_down_output_ptr_;        // [expert_num]

};

#endif