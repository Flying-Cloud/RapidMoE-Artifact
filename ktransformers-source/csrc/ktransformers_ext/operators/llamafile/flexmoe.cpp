/**
 * @Description  :
 * @Author       : wangwx19
 **/
#include "flexmoe.h"
#include <iostream>
#include <cstdint>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

FlexMOE::FlexMOE(FlexMOEConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;
    
    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    gate_proj_numa_.resize(numa_nodes);
    up_proj_numa_.resize(numa_nodes);
    down_proj_numa_.resize(numa_nodes);
    size_t exp_inter_hidden_mul_ = (size_t)config.expert_num * config.intermediate_size * config.hidden_size;
    
    size_t gate_numa_size = (size_t)config.expert_num * config.intermediate_size * ggml_row_size(config.gate_type,config.hidden_size);
    size_t up_numa_size = (size_t)config.expert_num * config.intermediate_size * ggml_row_size(config.up_type,config.hidden_size);
    size_t down_numa_size = (size_t)config.expert_num * config.hidden_size * ggml_row_size(config.down_type,config.intermediate_size);
    for (int i = 0; i < numa_nodes; i++) {
        // gate_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type), i);
        // up_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type), i);
        // down_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type), i);
        // row_size
        gate_proj_numa_[i] = numa_alloc_onnode(gate_numa_size, i);
        up_proj_numa_[i] = numa_alloc_onnode(up_numa_size, i);
        down_proj_numa_[i] = numa_alloc_onnode(down_numa_size, i);
        if (!gate_proj_numa_[i]) {
            std::cout << "Memory allocation failed for gate_proj_numa_ on node " << i << std::endl;
        }
        if (!up_proj_numa_[i]) {
            std::cout << "Memory allocation failed for up_proj_numa_ on node " << i << std::endl;
        }
        if (!down_proj_numa_[i]) {
            std::cout << "Memory allocation failed for down_proj_numa_ on node " << i << std::endl;
        }

        //memcpy(gate_proj_numa_[i], gate_proj_, exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type));
        //memcpy(up_proj_numa_[i], up_proj_, exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type));
        //memcpy(down_proj_numa_[i], down_proj_, exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type));
        
        memcpy(gate_proj_numa_[i], gate_proj_, gate_numa_size);
        memcpy(up_proj_numa_[i], up_proj_,up_numa_size);
        memcpy(down_proj_numa_[i], down_proj_, down_numa_size);
    }
    #endif

    std::vector<std::pair<void**, uint64_t>> s_mem_requests;
    s_mem_requests.push_back({(void**)&s_input_fp32_, sizeof(float) * config_.hidden_size});
    s_mem_requests.push_back({(void**)&s_gate_input_, ggml_row_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, config_.hidden_size)});
    s_mem_requests.push_back({(void**)&s_up_input_, ggml_row_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type, config_.hidden_size)});
    s_gate_output_.resize(config_.routed_expert_num);
    s_up_output_.resize(config_.routed_expert_num);
    s_intermediate_fp32_.resize(config_.routed_expert_num);
    s_down_input_.resize(config_.routed_expert_num);
    s_down_output_.resize(config_.routed_expert_num);
    s_local_gate_gpu_input_.resize(config_.flex_topk);
    s_local_up_gpu_input_.resize(config_.flex_topk);
    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_mem_requests.push_back({(void**)&s_gate_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_up_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_down_input_[i], ggml_row_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type, config_.intermediate_size)});
        s_mem_requests.push_back({(void**)&s_down_output_[i], sizeof(float) * config_.hidden_size});
    }
    for(int i = 0; i < config_.flex_topk; i++){
        s_mem_requests.push_back({(void**)&s_local_gate_gpu_input_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_local_up_gpu_input_[i], sizeof(float) * config_.intermediate_size});
    }
    s_mem_requests.push_back({(void**)&s_output_fp32_, sizeof(float) * config_.hidden_size});
    shared_mem_buffer.alloc(this, s_mem_requests);

    std::vector<std::pair<void**, uint64_t>> m_mem_requests;
    m_input_fp32_.resize(config_.group_max_len);
    m_gate_input_.resize(config_.group_max_len);
    m_up_input_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_input_fp32_[i], sizeof(float) * config_.hidden_size});
        m_mem_requests.push_back({(void**)&m_gate_input_[i], ggml_row_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, config_.hidden_size)});
        m_mem_requests.push_back({(void**)&m_up_input_[i], ggml_row_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type,config_.hidden_size)});
    }
    m_mem_requests.push_back({(void**)&m_local_gate_input_, config_.routed_expert_num * config_.group_max_len * ggml_row_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, config_.hidden_size)});
    m_mem_requests.push_back({(void**)&m_local_up_input_, config_.routed_expert_num * config_.group_max_len * ggml_row_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type, config_.hidden_size)});
    m_mem_requests.push_back({(void**)&m_local_gate_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_up_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_intermediate_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_down_input_, config_.routed_expert_num * config_.group_max_len * ggml_row_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type, config_.intermediate_size)});
    m_mem_requests.push_back({(void**)&m_local_down_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_output_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_output_fp32_[i], sizeof(float) * config_.hidden_size});
    }
    shared_mem_buffer.alloc(this, m_mem_requests);

    m_local_pos_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_local_pos_[i].resize(config_.routed_expert_num);
    }
    m_local_num_.resize(config_.expert_num);
    m_local_offset_.resize(config_.expert_num);
    m_local_gate_input_ptr_.resize(config_.expert_num);
    m_local_up_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_intermediate_fp32_ptr_.resize(config_.expert_num);
    m_local_down_input_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);
}

FlexMOE::~FlexMOE() {
    shared_mem_buffer.dealloc(this);

    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    size_t gate_numa_size = (size_t)config_.expert_num * config_.intermediate_size * ggml_row_size(config_.gate_type,config_.hidden_size);
    size_t up_numa_size = (size_t)config_.expert_num * config_.intermediate_size * ggml_row_size(config_.up_type,config_.hidden_size);
    size_t down_numa_size = (size_t)config_.expert_num * config_.hidden_size * ggml_row_size(config_.down_type,config_.intermediate_size);
    for (int i = 0; i < numa_nodes; i++) {
        //numa_free(gate_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type));
        //numa_free(up_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type));
        //numa_free(down_proj_numa_[i], config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type));
        numa_free(gate_proj_numa_[i], gate_numa_size);
        numa_free(up_proj_numa_[i], up_numa_size);
        numa_free(down_proj_numa_[i], down_numa_size);
    }
    #endif
}

void FlexMOE::warm_up(Backend* backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(ggml_row_size(config_.hidden_type, config_.hidden_size));
    std::vector<uint8_t> output(ggml_row_size(config_.hidden_type, config_.hidden_size));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
    for (int i = 0; i < config_.expert_num; i++) {
        uint64_t expert_ids = i;
        float weights = 0;
        forward_one(1, &expert_ids, &weights, input.data(), output.data(), backend);
    }
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}

void FlexMOE::forward_flex(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, Backend* backend) {
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
            gate_input_ptr = up_input_ptr = s_gate_input_;
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = s_gate_input_;
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(s_input_fp32_, s_up_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = s_up_input_;
            } else {
                up_input_ptr = input;
            }
        }
    }
    uint64_t offset = config_.intermediate_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
    
    for (int i = 0; i < k; i++) {
        void * gate_proj_gpu = (uint8_t*)w12_projs + i * offset * 2;
        void * up_proj_gpu   = (uint8_t*)w12_projs + i * offset * 2 + offset;
        to_float(gate_proj_gpu, s_local_gate_gpu_input_[i], config_.intermediate_size, config_.hidden_type);
        to_float(up_proj_gpu, s_local_up_gpu_input_[i], config_.intermediate_size, config_.hidden_type);
    }
    
    int nth = config_.intermediate_size / config_.stride;
    auto stride_A = ggml_row_size(config_.gate_type, config_.hidden_size);
    auto stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, config_.hidden_size);
    auto stride_C = config_.stride;
    auto ne00 = config_.hidden_size;
    auto row_size_1 = ggml_row_size(config_.gate_ori_type, ne00);
    auto row_size_2 = ggml_row_size(config_.gate_res_type, ne00);
    // w12_projs already contains the GPU-computed base projection.  The CPU
    // path must evaluate only the residual weights before the two are added.
    int  res  = 1;
    backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id % nth;
        
        #ifdef USE_NUMA
        //void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + expert_id * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * config_.stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        #else
        // to enable residual quant, ptr has to be modified with offset
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + expert_id * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * config_.stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        //void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * ggml_row_size(config_.gate_type, config_.hidden_size);
        // void* gate_proj_ptr = (uint8_t*)gate_proj_;
        #endif
        float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.stride;
        
        // todo: llamcafile_sgemm for residual only
        auto offset = config_.intermediate_size * row_size_1 + (row_size_2 - row_size_1) * ith * config_.stride;
        iqk_mul_mat_ik_offs(config_.stride, 1, ne00, config_.gate_type, gate_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, gate_input_ptr, stride_B,
            (float*)(gate_output_ptr), stride_C,0,1,offset,res);
        //llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT, 
        //    ith, config_.intermediate_size);
        // throw std::runtime_error("FlexMOE::forward_flex not implemented");
        #ifdef USE_NUMA
        //void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + expert_id * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * config_.stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + expert_id * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * config_.stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #endif

        float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.stride;

        // todo: llamafile_sgemm for residual only
        // 默认up_type与gate_type相同
        iqk_mul_mat_ik_offs(config_.stride, 1, ne00, config_.up_type, up_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.up_type).vec_dot_type, up_input_ptr, stride_B,
            (float*)(up_output_ptr), stride_C,0,1,offset,res);
        //llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            // printf("s_gate_output_[%d][%d] = %f\n", expert_idx, i, s_gate_output_[expert_idx][i]);
            s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i] + s_local_gate_gpu_input_[expert_idx][i]) * (s_up_output_[expert_idx][i] + s_local_up_gpu_input_[expert_idx][i]);
            //s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * (s_up_output_[expert_idx][i] );
        }

        if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
            float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx] + ith * config_.stride;
            void* down_input_ptr = s_down_input_[expert_idx] + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        for (int i = 0; i < k; i++) {
            from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }
    nth = config_.hidden_size / config_.stride;
    stride_A = ggml_row_size(config_.down_type, config_.intermediate_size);
    stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type, config_.intermediate_size);
    stride_C = config_.stride;
    ne00 = config_.intermediate_size;
    row_size_1 = ggml_row_size(config_.down_ori_type, ne00);
    row_size_2 = ggml_row_size(config_.down_res_type, ne00);
    res  = 0;
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_output_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];

            #ifdef USE_NUMA
            //void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + expert_id * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * config_.stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
            #else
            void* down_proj_ptr = (uint8_t*)down_proj_ + expert_id * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * config_.stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
            
            //void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_id * config_.hidden_size + ith * config_.stride) * ggml_row_size(config_.down_type, config_.intermediate_size);
            #endif
            auto proj_offset = (config_.down_type == config_.down_res_type) ? 
            config_.hidden_size * ggml_row_size(config_.down_ori_type, config_.intermediate_size)+ ith * config_.stride * (ggml_row_size(config_.down_type, config_.intermediate_size) - 2 * ggml_row_size(config_.down_ori_type, config_.intermediate_size)): 0;
            down_proj_ptr = (void*)((uint8_t*)down_proj_ptr + proj_offset);        
            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.stride;
            auto offset = config_.hidden_size * row_size_1 + (row_size_2 - row_size_1) * ith * config_.stride;
            iqk_mul_mat_ik_offs(config_.stride, 1, ne00, config_.down_type, down_proj_ptr, stride_A,
                ggml_internal_get_type_traits(config_.down_type).vec_dot_type, s_down_input_[expert_idx], stride_B,
                (float*)(down_output_ptr), stride_C,0,1,offset,res);
            //llamafile_sgemm(config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT,
            //    ith,config_.hidden_size);
            // llamafile_sgemm(config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_output_fp32_[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            float* output_fp32_ptr = s_output_fp32_ + ith * config_.stride;
            void* output_ptr = (uint8_t*)output + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
            from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(config_.hidden_type) != 0) {
        from_float(s_output_fp32_, output, config_.hidden_size, config_.hidden_type);
    }
}

void FlexMOE::forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
            gate_input_ptr = up_input_ptr = s_gate_input_;
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = s_gate_input_;
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(s_input_fp32_, s_up_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = s_up_input_;
            } else {
                up_input_ptr = input;
            }
        }
    }
    int nth = config_.intermediate_size / config_.stride;
    auto stride_A = ggml_row_size(config_.gate_type, config_.hidden_size);
    auto stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, config_.hidden_size);
    auto stride_C = config_.stride;
    auto ne00 = config_.hidden_size;
    auto row_size_1 = ggml_row_size(config_.gate_ori_type, ne00);
    auto row_size_2 = ggml_row_size(config_.gate_res_type, ne00);
    int  res  = 0;
    backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id % nth;
        
        #ifdef USE_NUMA
        // void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] +expert_id * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * config_.stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + expert_id * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * config_.stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        #endif

        float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.stride;
        // todo: llamafile_sgemm for residual only
        auto offset = config_.intermediate_size * row_size_1 + (row_size_2 - row_size_1) * ith * config_.stride;
        iqk_mul_mat_ik_offs(config_.stride, 1, ne00, config_.gate_type, gate_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, gate_input_ptr, stride_B,
            (float*)(gate_output_ptr), stride_C,0,1,offset,res);
        
        #ifdef USE_NUMA
        //void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + expert_id * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * config_.stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + expert_id * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * config_.stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #endif

        float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.stride;
        iqk_mul_mat_ik_offs(config_.stride, 1, ne00, config_.up_type, up_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.up_type).vec_dot_type, up_input_ptr, stride_B,
            (float*)(up_output_ptr), stride_C,0,1,offset,res);

        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
        }
        if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
            float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx] + ith * config_.stride;
            void* down_input_ptr = s_down_input_[expert_idx] + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        for (int i = 0; i < k; i++) {
            from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }
    nth = config_.hidden_size / config_.stride;
    stride_A = ggml_row_size(config_.down_type, config_.intermediate_size);
    stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type, config_.intermediate_size);
    stride_C = config_.stride;
    ne00 = config_.intermediate_size;
    row_size_1 = ggml_row_size(config_.down_ori_type, ne00);
    row_size_2 = ggml_row_size(config_.down_res_type, ne00);
    res  = 0;
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_output_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];

            #ifdef USE_NUMA
            //void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + expert_id * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * config_.stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
            #else
            void* down_proj_ptr = (uint8_t*)down_proj_ + expert_id * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * config_.stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
            #endif
            auto proj_offset = (config_.down_type == config_.down_res_type) ? 
            config_.hidden_size * ggml_row_size(config_.down_ori_type, config_.intermediate_size)+ ith * config_.stride * (ggml_row_size(config_.down_type, config_.intermediate_size) - 2 * ggml_row_size(config_.down_ori_type, config_.intermediate_size)): 0;
            down_proj_ptr = (void*)((uint8_t*)down_proj_ptr + proj_offset);
            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.stride;
            auto offset = config_.hidden_size * row_size_1 + (row_size_2 - row_size_1) * ith * config_.stride;
            iqk_mul_mat_ik_offs(config_.stride, 1, ne00, config_.down_type, down_proj_ptr, stride_A,
                ggml_internal_get_type_traits(config_.down_type).vec_dot_type, s_down_input_[expert_idx], stride_B,
                (float*)(down_output_ptr), stride_C,0,1,offset,res);

            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_output_fp32_[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            float* output_fp32_ptr = s_output_fp32_ + ith * config_.stride;
            void* output_ptr = (uint8_t*)output + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
            from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(config_.hidden_type) != 0) {
        from_float(s_output_fp32_, output, config_.hidden_size, config_.hidden_type);
    }
}

void FlexMOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
        m_local_down_input_ptr_[i] = m_local_down_input_ + offset * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
        offset += m_local_num_[i];
    }
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        const void* gate_input_ptr;
        const void* up_input_ptr;
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), gate_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), up_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        }
    }, nullptr);
    // wwx: QK_K 256 not fit for IQ1_S_R4 
    int stride = (ggml_internal_get_type_traits(config_.up_type).vec_dot_type == GGML_TYPE_Q8_K128) ? 128 : 256;
    int nth = config_.intermediate_size / stride;
    auto stride_A = ggml_row_size(config_.gate_type, config_.hidden_size);
    auto stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, config_.hidden_size);
    auto stride_C = config_.intermediate_size;
    auto ne00 = config_.hidden_size;
    auto row_size_1 = ggml_row_size(config_.gate_ori_type, ne00);
    auto row_size_2 = ggml_row_size(config_.gate_res_type, ne00);
    int  res  = 0;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        //void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + expert_idx * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        #else
        //void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * ggml_row_size(config_.gate_type, config_.hidden_size);
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + expert_idx * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        #endif

        float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        
        auto offset = config_.intermediate_size * row_size_1 + (row_size_2 - row_size_1) * ith * stride;
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.gate_type, gate_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, gate_input_ptr, stride_B,
            (float*)(gate_output_ptr), stride_C,0,1,offset,res);
        void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        //void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + expert_idx * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #else
        //void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * ggml_row_size(config_.gate_type, config_.hidden_size);
        void* up_proj_ptr = (uint8_t*)up_proj_ + expert_idx * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #endif

        float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.up_type, up_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.up_type).vec_dot_type, up_input_ptr, stride_B,
            (float*)(up_output_ptr), stride_C,0,1,offset,res);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) * m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
                //printf("m_local_intermediate_fp32_ptr_[%d][%d]:%f \n",expert_idx,i * config_.intermediate_size + j, m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j]);
            }
            // 打印m_local_intermediate_fp32_ptr_[expert_idx]
            //for (int j = 0; j < config_.intermediate_size; j++) {
            //    printf("DIn:%f ", m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j]);
            //}
            //printf("\n");
            float* intermediate_fp32_ptr = m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* down_input_ptr = m_local_down_input_ptr_[expert_idx] + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    stride = (ggml_internal_get_type_traits(config_.up_type).vec_dot_type == GGML_TYPE_Q8_K128) ? 128 : 256;
    nth = config_.hidden_size / stride;
    stride_A = ggml_row_size(config_.down_type, config_.intermediate_size);
    stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type, config_.intermediate_size);
    stride_C = config_.hidden_size;
    ne00 = config_.intermediate_size;
    row_size_1 = ggml_row_size(config_.down_ori_type, ne00);
    row_size_2 = ggml_row_size(config_.down_res_type, ne00);
    res  = 0;

    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* down_input_ptr = m_local_down_input_ptr_[expert_idx];
        #ifdef USE_NUMA
        //void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] +expert_idx * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
        #else
        //void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * ggml_row_size(config_.down_type, config_.intermediate_size);
        void* down_proj_ptr = (uint8_t*)down_proj_ + expert_idx * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
        #endif
        auto proj_offset = (config_.down_type == config_.down_res_type) ? 
        config_.hidden_size * ggml_row_size(config_.down_ori_type, config_.intermediate_size)+ ith * stride * (ggml_row_size(config_.down_type, config_.intermediate_size) - 2 * ggml_row_size(config_.down_ori_type, config_.intermediate_size)): 0;
        down_proj_ptr = (void*)((uint8_t*)down_proj_ptr + proj_offset);
        float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        auto offset = config_.hidden_size * row_size_1 + (row_size_2 - row_size_1) * ith * stride;
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.down_type, down_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.down_type).vec_dot_type, down_input_ptr, stride_B,
            (float*)(down_output_ptr), stride_C,0,1,offset,res);
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] * weights[i * k + j];
                //printf("m_local_down_output_ptr_[%d][%d]: %f\n", expert_ids[i * k + j],m_local_pos_[i][j] * config_.hidden_size + e,m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e]);
            }
        }
        from_float(m_output_fp32_[i], (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    }, nullptr);
}

void FlexMOE::forward_many_ftop(int qlen, int bsz, int k, const uint64_t* expert_ids, const float* weights, const void* input,const void* w12_projs, void* output, Backend* backend) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
        m_local_offset_[i] = 0;
    }
    for (int i = 0; i < bsz; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }
    for (int i = 1; i < config_.expert_num; i++) {
        m_local_offset_[i] = m_local_num_[i-1] + m_local_offset_[i-1];
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
        m_local_down_input_ptr_[i] = m_local_down_input_ + offset * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
        offset += m_local_num_[i];
    }
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        const void* gate_input_ptr;
        const void* up_input_ptr;
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), gate_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), up_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        }
        /*
        uint64_t offset_w12 = config_.intermediate_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        for (int j = 0; j < k; j++){
            // w12中已按照sorted_token_ids进行排序
            void * gate_proj_gpu = (uint8_t*)w12_projs + m_local_offset[expert_ids[i*k + j]] * offset_w12 * 2 + m_local_pos_[i][j] * offset_w12 * 2;
            void * up_proj_gpu   = (uint8_t*)w12_projs + m_local_offset[expert_ids[i*k + j]] * offset_w12 * 2 + m_local_pos_[i][j] * offset_w12 * 2 + offset_w12;
            // 此处*2是因为FP32是BF16的两倍，因此默认hidden_type为BF16
            to_float(gate_proj_gpu, (float*)m_local_gate_gpu_input_[expert_ids[i*k + j]] + m_local_pos[i][j] * config_.intermediate_size, config_.intermediate_size, config_.hidden_type);
            to_float(up_proj_gpu,   (float*)m_local_up_gpu_input_[expert_ids[i*k + j]] + m_local_pos[i][j] * config_.intermediate_size, config_.intermediate_size, config_.hidden_type);
        }
        */
    }, nullptr);
    // wwx: QK_K 256 not fit for IQ1_S_R4 
    int stride = (ggml_internal_get_type_traits(config_.up_type).vec_dot_type == GGML_TYPE_Q8_K128) ? 128 : 256;
    int nth = config_.intermediate_size / stride;
    auto stride_A = ggml_row_size(config_.gate_type, config_.hidden_size);
    auto stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, config_.hidden_size);
    auto stride_C = config_.intermediate_size;
    auto ne00 = config_.hidden_size;
    auto row_size_1 = ggml_row_size(config_.gate_ori_type, ne00);
    auto row_size_2 = ggml_row_size(config_.gate_res_type, ne00);
    int  res  = 1;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        //void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + expert_idx * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        #else
        //void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * ggml_row_size(config_.gate_type, config_.hidden_size);
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + expert_idx * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        #endif

        float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        
        auto offset = config_.intermediate_size * row_size_1 + (row_size_2 - row_size_1) * ith * stride;
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.gate_type, gate_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, gate_input_ptr, stride_B,
            (float*)(gate_output_ptr), stride_C,0,1,offset,res);
        void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        //void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + expert_idx * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #else
        //void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * ggml_row_size(config_.gate_type, config_.hidden_size);
        void* up_proj_ptr = (uint8_t*)up_proj_ + expert_idx * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #endif

        float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.up_type, up_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.up_type).vec_dot_type, up_input_ptr, stride_B,
            (float*)(up_output_ptr), stride_C,0,1,offset,res);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                float* gate_gpu_ptr_ = (float*)w12_projs + m_local_offset_[expert_idx] * config_.intermediate_size * 2 + i * config_.intermediate_size * 2;
                float* up_gpu_ptr_   = (float*)w12_projs + m_local_offset_[expert_idx] * config_.intermediate_size * 2 + i * config_.intermediate_size * 2 + config_.intermediate_size;
                m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j] + gate_gpu_ptr_[j]) * (m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j] + up_gpu_ptr_[j]);
                //printf("m_local_intermediate_fp32_ptr_[%d][%d]:%f \n",expert_idx,i * config_.intermediate_size + j, m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j]);
            }
            // 打印m_local_intermediate_fp32_ptr_[expert_idx]
            //for (int j = 0; j < config_.intermediate_size; j++) {
            //    printf("DIn:%f ", m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j]);
            //}
            //printf("\n");
            float* intermediate_fp32_ptr = m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* down_input_ptr = m_local_down_input_ptr_[expert_idx] + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    stride = (ggml_internal_get_type_traits(config_.down_type).vec_dot_type == GGML_TYPE_Q8_K128) ? 128 : 256;
    nth = config_.hidden_size / stride;
    stride_A = ggml_row_size(config_.down_type, config_.intermediate_size);
    stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type, config_.intermediate_size);
    stride_C = config_.hidden_size;
    ne00 = config_.intermediate_size;
    row_size_1 = ggml_row_size(config_.down_ori_type, ne00);
    row_size_2 = ggml_row_size(config_.down_res_type, ne00);
    res  = 0;

    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* down_input_ptr = m_local_down_input_ptr_[expert_idx];
        #ifdef USE_NUMA
        //void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] +expert_idx * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
        #else
        //void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * ggml_row_size(config_.down_type, config_.intermediate_size);
        void* down_proj_ptr = (uint8_t*)down_proj_ + expert_idx * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
        #endif
        auto proj_offset = (config_.down_type == config_.down_res_type) ? 
        config_.hidden_size * ggml_row_size(config_.down_ori_type, config_.intermediate_size)+ ith * stride * (ggml_row_size(config_.down_type, config_.intermediate_size) - 2 * ggml_row_size(config_.down_ori_type, config_.intermediate_size)): 0;
        down_proj_ptr = (void*)((uint8_t*)down_proj_ptr + proj_offset);
        float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        auto offset = config_.hidden_size * row_size_1 + (row_size_2 - row_size_1) * ith * stride;
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.down_type, down_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.down_type).vec_dot_type, down_input_ptr, stride_B,
            (float*)(down_output_ptr), stride_C,0,1,offset,res);
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] * weights[i * k + j];
                //printf("m_local_down_output_ptr_[%d][%d]: %f\n", expert_ids[i * k + j],m_local_pos_[i][j] * config_.hidden_size + e,m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e]);
            }
        }
        from_float(m_output_fp32_[i], (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    }, nullptr);
}


void FlexMOE::forward_many_ftop_v1(int qlen, int bsz, int k, int max_k, const uint64_t* expert_ids, const float* weights, const void* input,const void* w12_projs, void* output, Backend* backend) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
        m_local_offset_[i] = 0;
    }
    for (int i = 0; i < bsz; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * max_k + j]]++;
        }
    }
    for (int i = 1; i < config_.expert_num; i++) {
        m_local_offset_[i] = m_local_num_[i-1] + m_local_offset_[i-1];
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
        m_local_down_input_ptr_[i] = m_local_down_input_ + offset * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
        offset += m_local_num_[i];
    }
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        const void* gate_input_ptr;
        const void* up_input_ptr;
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * max_k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), gate_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * max_k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), up_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        }
        /*
        uint64_t offset_w12 = config_.intermediate_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        for (int j = 0; j < k; j++){
            // w12中已按照sorted_token_ids进行排序
            void * gate_proj_gpu = (uint8_t*)w12_projs + m_local_offset[expert_ids[i*k + j]] * offset_w12 * 2 + m_local_pos_[i][j] * offset_w12 * 2;
            void * up_proj_gpu   = (uint8_t*)w12_projs + m_local_offset[expert_ids[i*k + j]] * offset_w12 * 2 + m_local_pos_[i][j] * offset_w12 * 2 + offset_w12;
            // 此处*2是因为FP32是BF16的两倍，因此默认hidden_type为BF16
            to_float(gate_proj_gpu, (float*)m_local_gate_gpu_input_[expert_ids[i*k + j]] + m_local_pos[i][j] * config_.intermediate_size, config_.intermediate_size, config_.hidden_type);
            to_float(up_proj_gpu,   (float*)m_local_up_gpu_input_[expert_ids[i*k + j]] + m_local_pos[i][j] * config_.intermediate_size, config_.intermediate_size, config_.hidden_type);
        }
        */
    }, nullptr);
    // wwx: QK_K 256 not fit for IQ1_S_R4 
    int stride = (ggml_internal_get_type_traits(config_.up_type).vec_dot_type == GGML_TYPE_Q8_K128) ? 128 : 256;
    int nth = config_.intermediate_size / stride;
    auto stride_A = ggml_row_size(config_.gate_type, config_.hidden_size);
    auto stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, config_.hidden_size);
    auto stride_C = config_.intermediate_size;
    auto ne00 = config_.hidden_size;
    auto row_size_1 = ggml_row_size(config_.gate_ori_type, ne00);
    auto row_size_2 = ggml_row_size(config_.gate_res_type, ne00);
    int  res  = 1;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        //void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + expert_idx * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        #else
        //void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * ggml_row_size(config_.gate_type, config_.hidden_size);
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + expert_idx * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        #endif

        float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        
        auto offset = config_.intermediate_size * row_size_1 + (row_size_2 - row_size_1) * ith * stride;
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.gate_type, gate_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, gate_input_ptr, stride_B,
            (float*)(gate_output_ptr), stride_C,0,1,offset,res);
        void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        //void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + expert_idx * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #else
        //void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * ggml_row_size(config_.gate_type, config_.hidden_size);
        void* up_proj_ptr = (uint8_t*)up_proj_ + expert_idx * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #endif

        float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.up_type, up_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.up_type).vec_dot_type, up_input_ptr, stride_B,
            (float*)(up_output_ptr), stride_C,0,1,offset,res);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                float* gate_gpu_ptr_ = (float*)w12_projs + m_local_offset_[expert_idx] * config_.intermediate_size * 2 + i * config_.intermediate_size * 2;
                float* up_gpu_ptr_   = (float*)w12_projs + m_local_offset_[expert_idx] * config_.intermediate_size * 2 + i * config_.intermediate_size * 2 + config_.intermediate_size;
                m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j] + gate_gpu_ptr_[j]) * (m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j] + up_gpu_ptr_[j]);
                //printf("m_local_intermediate_fp32_ptr_[%d][%d]:%f \n",expert_idx,i * config_.intermediate_size + j, m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j]);
            }
            // 打印m_local_intermediate_fp32_ptr_[expert_idx]
            //for (int j = 0; j < config_.intermediate_size; j++) {
            //    printf("DIn:%f ", m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j]);
            //}
            //printf("\n");
            float* intermediate_fp32_ptr = m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* down_input_ptr = m_local_down_input_ptr_[expert_idx] + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    stride = (ggml_internal_get_type_traits(config_.down_type).vec_dot_type == GGML_TYPE_Q8_K128) ? 128 : 256;
    nth = config_.hidden_size / stride;
    stride_A = ggml_row_size(config_.down_type, config_.intermediate_size);
    stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type, config_.intermediate_size);
    stride_C = config_.hidden_size;
    ne00 = config_.intermediate_size;
    row_size_1 = ggml_row_size(config_.down_ori_type, ne00);
    row_size_2 = ggml_row_size(config_.down_res_type, ne00);
    res  = 0;

    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* down_input_ptr = m_local_down_input_ptr_[expert_idx];
        #ifdef USE_NUMA
        //void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] +expert_idx * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
        #else
        //void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * ggml_row_size(config_.down_type, config_.intermediate_size);
        void* down_proj_ptr = (uint8_t*)down_proj_ + expert_idx * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
        #endif
        auto proj_offset = (config_.down_type == config_.down_res_type) ? 
        config_.hidden_size * ggml_row_size(config_.down_ori_type, config_.intermediate_size)+ ith * stride * (ggml_row_size(config_.down_type, config_.intermediate_size) - 2 * ggml_row_size(config_.down_ori_type, config_.intermediate_size)): 0;
        down_proj_ptr = (void*)((uint8_t*)down_proj_ptr + proj_offset);
        float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        auto offset = config_.hidden_size * row_size_1 + (row_size_2 - row_size_1) * ith * stride;
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.down_type, down_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.down_type).vec_dot_type, down_input_ptr, stride_B,
            (float*)(down_output_ptr), stride_C,0,1,offset,res);
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_ptr_[expert_ids[i * max_k + j]][m_local_pos_[i][j] * config_.hidden_size + e] * weights[i * max_k + j];
                //printf("m_local_down_output_ptr_[%d][%d]: %f\n", expert_ids[i * k + j],m_local_pos_[i][j] * config_.hidden_size + e,m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e]);
            }
        }
        from_float(m_output_fp32_[i], (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    }, nullptr);
}

void FlexMOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, int* batch_size_tensor, Backend* backend) {
    qlen = batch_size_tensor[0];
    if (qlen < config_.group_min_len) {
        for (int i = 0; i < qlen; i++) {
            forward_one(k, expert_ids + i * k, weights + i * k, (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
        }
        return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);
    forward_many(forward_len, k, expert_ids, weights, input, output, backend);

    batch_size_tensor[0] -= forward_len;
    forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k, (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), batch_size_tensor, backend);
}

void FlexMOE::forward_many_flex(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, int* batch_size_tensor, Backend* backend) {
    int qlen1 = batch_size_tensor[0];
    int forward_len = std::min(config_.group_max_len, qlen1);
    forward_many_ftop(forward_len, qlen, k, expert_ids, weights, input, w12_projs, output, backend);
    return;
}

void FlexMOE::forward_many_flex_v1(int qlen, const uint64_t* k, int max_k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, int* batch_size_tensor, Backend* backend) {
    int k1 = k[0];
    int qlen1 = batch_size_tensor[0];
    int forward_len = std::min(config_.group_max_len, qlen1);
    if (k1 == 0) return;
    forward_many_ftop_v1(forward_len, qlen, k1, max_k, expert_ids, weights, input, w12_projs, output, backend);
    //forward_many_v1(forward_len, qlen, k1, max_k, expert_ids, weights, input, output, backend);
    return;
}

void FlexMOE::forward_many_v1(int qlen, int bsz, int k, int max_k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
        m_local_offset_[i] = 0;
    }
    for (int i = 0; i < bsz; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * max_k + j]]++;
        }
    }
    for (int i = 1; i < config_.expert_num; i++) {
        m_local_offset_[i] = m_local_num_[i-1] + m_local_offset_[i-1];
    }

    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
        m_local_down_input_ptr_[i] = m_local_down_input_ + offset * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
        offset += m_local_num_[i];
    }
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        const void* gate_input_ptr;
        const void* up_input_ptr;
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * max_k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), gate_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * max_k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), up_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        }
    }, nullptr);
    // wwx: QK_K 256 not fit for IQ1_S_R4 
    int stride = (ggml_internal_get_type_traits(config_.up_type).vec_dot_type == GGML_TYPE_Q8_K128) ? 128 : 256;
    int nth = config_.intermediate_size / stride;
    auto stride_A = ggml_row_size(config_.gate_type, config_.hidden_size);
    auto stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, config_.hidden_size);
    auto stride_C = config_.intermediate_size;
    auto ne00 = config_.hidden_size;
    auto row_size_1 = ggml_row_size(config_.gate_ori_type, ne00);
    auto row_size_2 = ggml_row_size(config_.gate_res_type, ne00);
    int  res  = 0;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        //void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + expert_idx * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        #else
        //void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * ggml_row_size(config_.gate_type, config_.hidden_size);
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + expert_idx * config_.intermediate_size * ggml_row_size(config_.gate_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.gate_ori_type, config_.hidden_size);
        #endif

        float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        
        auto offset = config_.intermediate_size * row_size_1 + (row_size_2 - row_size_1) * ith * stride;
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.gate_type, gate_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, gate_input_ptr, stride_B,
            (float*)(gate_output_ptr), stride_C,0,1,offset,res);
        void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        //void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + expert_idx * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #else
        //void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * ggml_row_size(config_.gate_type, config_.hidden_size);
        void* up_proj_ptr = (uint8_t*)up_proj_ + expert_idx * config_.intermediate_size * ggml_row_size(config_.up_type, config_.hidden_size) + ith * stride * ggml_row_size(config_.up_ori_type, config_.hidden_size);
        #endif

        float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.up_type, up_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.up_type).vec_dot_type, up_input_ptr, stride_B,
            (float*)(up_output_ptr), stride_C,0,1,offset,res);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) * m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
                //printf("m_local_intermediate_fp32_ptr_[%d][%d]:%f \n",expert_idx,i * config_.intermediate_size + j, m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j]);
            }
            // 打印m_local_intermediate_fp32_ptr_[expert_idx]
            //for (int j = 0; j < config_.intermediate_size; j++) {
            //    printf("DIn:%f ", m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j]);
            //}
            //printf("\n");
            float* intermediate_fp32_ptr = m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* down_input_ptr = m_local_down_input_ptr_[expert_idx] + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    stride = (ggml_internal_get_type_traits(config_.up_type).vec_dot_type == GGML_TYPE_Q8_K128) ? 128 : 256;
    nth = config_.hidden_size / stride;
    stride_A = ggml_row_size(config_.down_type, config_.intermediate_size);
    stride_B = ggml_row_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type, config_.intermediate_size);
    stride_C = config_.hidden_size;
    ne00 = config_.intermediate_size;
    row_size_1 = ggml_row_size(config_.down_ori_type, ne00);
    row_size_2 = ggml_row_size(config_.down_res_type, ne00);
    res  = 0;

    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* down_input_ptr = m_local_down_input_ptr_[expert_idx];
        #ifdef USE_NUMA
        //void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] +expert_idx * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
        #else
        //void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * ggml_row_size(config_.down_type, config_.intermediate_size);
        void* down_proj_ptr = (uint8_t*)down_proj_ + expert_idx * config_.hidden_size * ggml_row_size(config_.down_type, config_.intermediate_size) + ith * stride * ggml_row_size(config_.down_ori_type, config_.intermediate_size);
        #endif
        auto proj_offset = (config_.down_type == config_.down_res_type) ? 
        config_.hidden_size * ggml_row_size(config_.down_ori_type, config_.intermediate_size)+ ith * stride * (ggml_row_size(config_.down_type, config_.intermediate_size) - 2 * ggml_row_size(config_.down_ori_type, config_.intermediate_size)): 0;
        down_proj_ptr = (void*)((uint8_t*)down_proj_ptr + proj_offset);
        float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * stride;
        //llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        auto offset = config_.hidden_size * row_size_1 + (row_size_2 - row_size_1) * ith * stride;
        iqk_mul_mat_ik_offs(stride, m_local_num_[expert_idx], ne00, config_.down_type, down_proj_ptr, stride_A,
            ggml_internal_get_type_traits(config_.down_type).vec_dot_type, down_input_ptr, stride_B,
            (float*)(down_output_ptr), stride_C,0,1,offset,res);
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_ptr_[expert_ids[i * max_k + j]][m_local_pos_[i][j] * config_.hidden_size + e] * weights[i * max_k + j];
                //printf("m_local_down_output_ptr_[%d][%d]: %f\n", expert_ids[i * k + j],m_local_pos_[i][j] * config_.hidden_size + e,m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e]);
            }
        }
        from_float(m_output_fp32_[i], (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    }, nullptr);
}

void FlexMOE::forward_flex_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, int* batch_size_tensor, Backend* backend) {
    qlen = batch_size_tensor[0];
    uint64_t offset = k * 2 * config_.intermediate_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
    for(int i = 0; i < qlen; i++){
        forward_flex(1, k, expert_ids + i * k, weights + i * k, (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type),(uint8_t*) w12_projs + i * offset ,(uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
    }
}

void FlexMOE::forward_flex_many_v1(int qlen, const uint64_t* k, int max_k, const uint64_t* expert_ids, const float* weights, const void* input, const void* w12_projs, void* output, int* batch_size_tensor, Backend* backend) {
    int k1 = k[0];
    qlen = batch_size_tensor[0];
    if (k1 == 0) return;
    uint64_t offset = k1 * 2 * config_.intermediate_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
    for(int i = 0; i < qlen; i++){
        forward_flex(1, k1, expert_ids + i * max_k, weights + i * max_k, (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type),(uint8_t*) w12_projs + i * offset ,(uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend);
    }
}
