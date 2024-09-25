#pragma once
#include <iostream>
#include <map>
#include <string>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "utils.hpp"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#define MAX_PARAMS_TENSOR_NUM 15360
#define MAX_GRAPH_SIZE 15360
#define MAX_DIMS 5

struct TensorStorage {
    std::string name;
    ggml_type type       = GGML_TYPE_F32;
    int64_t ne[MAX_DIMS] = {1, 1, 1, 1, 1};
    int n_dims           = 0;

    size_t offset = 0;  // offset in file

    TensorStorage() = default;

    TensorStorage(const std::string& name, ggml_type type, int64_t* ne, int n_dims, size_t offset = 0)
        : name(name), type(type), n_dims(n_dims), offset(offset) {
        for (int i = 0; i < n_dims; i++) {
            this->ne[i] = ne[i];
        }
    }

    int64_t nelements() const {
        int64_t n = 1;
        for (int i = 0; i < MAX_DIMS; i++) {
            n *= ne[i];
        }
        return n;
    }

    int64_t nbytes() const {
        return nelements() * ggml_type_size(type) / ggml_blck_size(type);
    }

    int64_t nbytes_to_read() const {
        return nbytes();
    }

    void unsqueeze() {
        if (n_dims == 2) {
            n_dims = 4;
            ne[3]  = ne[1];
            ne[2]  = ne[0];
            ne[1]  = 1;
            ne[0]  = 1;
        }
    }

    std::vector<TensorStorage> chunk(size_t n) {
        std::vector<TensorStorage> chunks;
        size_t chunk_size = nbytes_to_read() / n;
        // printf("%d/%d\n", chunk_size, nbytes_to_read());
        reverse_ne();
        for (int i = 0; i < n; i++) {
            TensorStorage chunk_i = *this;
            chunk_i.ne[0]         = ne[0] / n;
            chunk_i.offset        = offset + i * chunk_size;
            chunk_i.reverse_ne();
            chunks.push_back(chunk_i);
        }
        reverse_ne();
        return chunks;
    }

    void reverse_ne() {
        int64_t new_ne[MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) {
            new_ne[i] = ne[n_dims - 1 - i];
        }
        for (int i = 0; i < n_dims; i++) {
            ne[i] = new_ne[i];
        }
    }

    std::string to_string() const {
        std::stringstream ss;
        const char* type_name = ggml_type_name(type);
        ss << name << " | " << type_name << " | ";
        ss << n_dims << " [";
        for (int i = 0; i < MAX_DIMS; i++) {
            ss << ne[i];
            if (i != MAX_DIMS - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }
};

struct GGMLRunner {
protected:
    typedef std::function<struct ggml_cgraph*()> get_graph_cb_t;

    struct ggml_context* params_ctx     = NULL;
    ggml_backend_buffer_t params_buffer = NULL;

    struct ggml_context* compute_ctx    = NULL;
    struct ggml_gallocr* compute_allocr = NULL;

    std::map<struct ggml_tensor*, const void*> backend_tensor_data_map;

    ggml_type wtype        = GGML_TYPE_F32;
    ggml_backend_t backend = NULL;

    void alloc_params_ctx() {
        struct ggml_init_params params;
        params.mem_size =
            static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = NULL;
        params.no_alloc   = true;

        params_ctx = ggml_init(params);
        GGML_ASSERT(params_ctx != NULL);
    }

    void free_params_ctx() {
        if (params_ctx != NULL) {
            ggml_free(params_ctx);
            params_ctx = NULL;
        }
    }

    void alloc_compute_ctx() {
        struct ggml_init_params params;
        params.mem_size = static_cast<size_t>(
            ggml_tensor_overhead() * MAX_GRAPH_SIZE + ggml_graph_overhead());
        params.mem_buffer = NULL;
        params.no_alloc   = true;

        compute_ctx = ggml_init(params);
        GGML_ASSERT(compute_ctx != NULL);
    }

    void free_compute_ctx() {
        if (compute_ctx != NULL) {
            ggml_free(compute_ctx);
            compute_ctx = NULL;
        }
    }

    bool alloc_compute_buffer(get_graph_cb_t get_graph) {
        if (compute_allocr != NULL) {
            return true;
        }
        reset_compute_ctx();
        struct ggml_cgraph* gf = get_graph();
        backend_tensor_data_map.clear();
        compute_allocr =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        if (!ggml_gallocr_reserve(compute_allocr, gf)) {
            printf("%s: failed to allocate the compute buffer\n", get_desc().c_str());
            free_compute_buffer();
            return false;
        }

        // compute the required memory
        size_t compute_buffer_size =
            ggml_gallocr_get_buffer_size(compute_allocr, 0);
        printf("%s compute buffer size: %.2f MB(%s)\n", get_desc().c_str(),
               compute_buffer_size / 1024.0 / 1024.0,
               ggml_backend_is_cpu(backend) ? "RAM" : "VRAM");
        return true;
    }

    void cpy_data_to_backend_tensor() {
        for (auto& kv : backend_tensor_data_map) {
            auto tensor = kv.first;
            auto data   = kv.second;

            ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
        }

        backend_tensor_data_map.clear();
    }

public:
    virtual std::string get_desc() = 0;

    GGMLRunner(ggml_backend_t backend, ggml_type wtype = GGML_TYPE_F32)
        : backend(backend), wtype(wtype) {
        alloc_params_ctx();
    }

    virtual ~GGMLRunner() {
        free_params_buffer();
        free_compute_buffer();
        free_params_ctx();
        free_compute_ctx();
    }

    void reset_compute_ctx() {
        free_compute_ctx();
        alloc_compute_ctx();
    }

    bool alloc_params_buffer() {
        int num_tensors = (int)ggml_tensor_num(params_ctx);
        params_buffer   = ggml_backend_alloc_ctx_tensors(params_ctx, backend);
        if (params_buffer == NULL) {
            printf("%s alloc params backend buffer failed, num_tensors = %d\n",
                   get_desc().c_str(), num_tensors);
            return false;
        }
        size_t params_buffer_size = ggml_backend_buffer_get_size(params_buffer);
        printf("%s params backend buffer size = % 6.2f MB(%s) (%d tensors)\n",
               get_desc().c_str(), params_buffer_size / (1024.0 * 1024.0),
               ggml_backend_is_cpu(backend) ? "RAM" : "VRAM", num_tensors);
        return true;
    }

    void free_params_buffer() {
        if (params_buffer != NULL) {
            ggml_backend_buffer_free(params_buffer);
            params_buffer = NULL;
        }
    }

    size_t get_params_buffer_size() {
        if (params_buffer != NULL) {
            return ggml_backend_buffer_get_size(params_buffer);
        }
        return 0;
    }

    void free_compute_buffer() {
        if (compute_allocr != NULL) {
            ggml_gallocr_free(compute_allocr);
            compute_allocr = NULL;
        }
    }

    // do copy after alloc graph
    void set_backend_tensor_data(struct ggml_tensor* tensor, const void* data) {
        backend_tensor_data_map[tensor] = data;
    }

    struct ggml_tensor* to_backend(struct ggml_tensor* tensor) {
        GGML_ASSERT(compute_ctx != NULL);
        if (tensor == NULL) {
            return NULL;
        }
        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(backend) &&
            (tensor->buffer == NULL ||
             ggml_backend_buffer_is_host(tensor->buffer))) {
            // pass input tensors to gpu memory
            auto backend_tensor = ggml_dup_tensor(compute_ctx, tensor);

            set_backend_tensor_data(backend_tensor, tensor->data);
            return backend_tensor;
        } else {
            return tensor;
        }
    }

    void compute(get_graph_cb_t get_graph, int n_threads, bool free_compute_buffer_immediately = true, struct ggml_tensor** output = NULL, struct ggml_context* output_ctx = NULL) {
        alloc_compute_buffer(get_graph);
        reset_compute_ctx();
        struct ggml_cgraph* gf = get_graph();
        GGML_ASSERT(ggml_gallocr_alloc_graph(compute_allocr, gf));
        cpy_data_to_backend_tensor();
        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }
#ifdef GGML_USE_METAL
        if (ggml_backend_is_metal(backend)) {
            ggml_backend_metal_set_n_cb(backend, n_threads);
        }
#endif
        ggml_backend_graph_compute(backend, gf);

#ifdef GGML_PERF
        ggml_graph_print(gf);
#endif
        if (output != NULL) {
            auto result = gf->nodes[gf->n_nodes - 1];
            if (*output == NULL && output_ctx != NULL) {
                *output = ggml_dup_tensor(output_ctx, result);
            }
            if (*output != NULL) {
                ggml_backend_tensor_get_and_sync(backend, result, (*output)->data, 0,
                                                 ggml_nbytes(*output));

                // ggml_backend_tensor_get(result, (*output)->data, 0,
                //                         ggml_nbytes(*output));
            }
        }

        if (free_compute_buffer_immediately) {
            free_compute_buffer();
        }
    }
};

class GGMLBlock {
protected:
    typedef std::unordered_map<std::string, struct ggml_tensor*> ParameterMap;
    typedef std::unordered_map<std::string, std::shared_ptr<GGMLBlock>>
        GGMLBlockMap;
    GGMLBlockMap blocks;
    ParameterMap params;

    virtual ~GGMLBlock(){};

    void init_blocks(struct ggml_context* ctx, ggml_type wtype) {
        for (auto& pair : blocks) {
            auto& block = pair.second;

            block->init(ctx, wtype);
        }
    }

    virtual void init_params(struct ggml_context* ctx, ggml_type wtype) {}

public:
    void init(struct ggml_context* ctx, ggml_type wtype) {
        init_blocks(ctx, wtype);
        init_params(ctx, wtype);
    }

    size_t get_params_num() {
        size_t num_tensors = params.size();
        for (auto& pair : blocks) {
            auto& block = pair.second;

            num_tensors += block->get_params_num();
        }
        return num_tensors;
    };

    size_t get_params_mem_size() {
        size_t mem_size = 0;
        for (auto& pair : blocks) {
            auto& block = pair.second;

            mem_size += block->get_params_mem_size();
        }

        for (auto& pair : params) {
            mem_size += ggml_nbytes(pair.second);
        }

        return mem_size;
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors,
                           std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }

        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->get_param_tensors(tensors, prefix + pair.first);
        }

        for (auto& pair : params) {
            struct ggml_tensor* param    = pair.second;
            tensors[prefix + pair.first] = pair.second;
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x);
};
