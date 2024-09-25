#pragma once
#include <iostream>
#include <map>
#include <string>
#include "common.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "model_config.hpp"
#include "utils.hpp"

class ModelLoader {
public:
    ModelLoader(CLIPVersion version_)
        : version(version_){};
    bool init_gguf_file(std::string file_path, const std::string& prefix = "");
    bool load_tensor(std::map<std::string, struct ggml_tensor*>& dst_tensors, ggml_backend_t backend);
    ~ModelLoader(){};

    std::vector<TensorStorage> tensor_storages;
    void convert_tensor(void* src,
                        ggml_type src_type,
                        void* dst,
                        ggml_type dst_type,
                        int nrows,
                        int n_per_row);

private:
    void pre_process_tensors(std::string& name);
    CLIPVersion version;
    std::string file_path_;
};

bool ModelLoader::init_gguf_file(std::string file_path, const std::string& prefix) {
    file_path_              = file_path;
    gguf_context* ctx_gguf_ = NULL;
    ggml_context* ctx_meta_ = NULL;

    struct gguf_init_params param = {true, &ctx_meta_};

    ctx_gguf_ = gguf_init_from_file(file_path_.c_str(), param);

    if (!ctx_gguf_) {
        printf("failed to open %s\n", file_path.c_str());
        return false;
    }

    int n_tensors = gguf_get_n_tensors(ctx_gguf_);

    size_t total_size  = 0;
    size_t data_offset = gguf_get_data_offset(ctx_gguf_);
    for (int i = 0; i < n_tensors; i++) {
        std::string name          = gguf_get_tensor_name(ctx_gguf_, i);
        struct ggml_tensor* dummy = ggml_get_tensor(ctx_meta_, name.c_str());
        size_t offset             = data_offset + gguf_get_tensor_offset(ctx_gguf_, i);

        pre_process_tensors(name);

        TensorStorage tensor_storage(prefix + name, dummy->type, dummy->ne, ggml_n_dims(dummy), offset);

        GGML_ASSERT(ggml_nbytes(dummy) == tensor_storage.nbytes());

        tensor_storages.push_back(tensor_storage);
    }

    gguf_free(ctx_gguf_);
    ggml_free(ctx_meta_);

    return true;
}

bool ModelLoader::load_tensor(std::map<std::string, struct ggml_tensor*>& tensors, ggml_backend_t backend) {
    std::ifstream fin(file_path_, std::ios::binary);

    if (!fin.is_open()) {
        fprintf(stderr, "failed to open %s\n", file_path_.c_str());
        return false;
    }

    std::vector<char> read_buf;
    std::vector<char> convert_buf;
    for (int i = 0; i < tensor_storages.size(); i++) {
        std::string name = tensor_storages[i].name;

        if (tensors.find(name) != tensors.end()) {
            std::string tensor_name      = name;
            struct ggml_tensor* tensor_v = tensors[name];

            int ne00 = tensor_v->ne[0], ne10 = tensor_storages[i].ne[0];
            int ne01 = tensor_v->ne[1], ne11 = tensor_storages[i].ne[1];
            int ne02 = tensor_v->ne[2], ne12 = tensor_storages[i].ne[2];
            int ne03 = tensor_v->ne[3], ne13 = tensor_storages[i].ne[3];

            if ((ne00 != ne10) || (ne01 != ne11) || (ne02 != ne12) || (ne03 != ne13)) {
                fprintf(stderr, "%s:%s(%dx%dx%dx%d) and %s(%dx%dx%dx%d) shape not same\n", __func__,
                        name.c_str(), ne00, ne01, ne02, ne03, tensor_storages[i].name.c_str(), ne10, ne11, ne12, ne13);
                return false;
            }

            int nelements    = ggml_nelements(tensor_v);
            const size_t bpe = ggml_type_size(ggml_type(tensor_v->type));
            if ((nelements * bpe) / ggml_blck_size(tensor_v->type) != ggml_nbytes(tensor_v)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.c_str(), ggml_nbytes(tensor_v), nelements * bpe);
                return false;
            }

            size_t offset = tensor_storages[i].offset;
            fin.seekg(offset);
            read_buf.clear();
            read_buf.resize(tensor_storages[i].nbytes());
            fin.read(read_buf.data(), tensor_storages[i].nbytes());

            if (ggml_backend_is_cpu(backend)) {
                convert_tensor(read_buf.data(), tensor_storages[i].type, tensor_v->data,
                               tensor_v->type, (int)tensor_storages[i].nelements() / (int)tensor_storages[i].ne[0],
                               (int)tensor_storages[i].ne[0]);
            } else {
                if (tensor_storages[i].type == tensor_v->type) {
                    ggml_backend_tensor_set(tensor_v, read_buf.data(), 0, ggml_nbytes(tensor_v));
                } else {
                    convert_buf.clear();
                    convert_buf.resize(ggml_nbytes(tensor_v));
                    convert_tensor((void*)read_buf.data(), tensor_storages[i].type, convert_buf.data(),
                                   tensor_v->type, (int)tensor_storages[i].nelements() / (int)tensor_storages[i].ne[0],
                                   (int)tensor_storages[i].ne[0]);
                    ggml_backend_tensor_set(tensor_v, convert_buf.data(), 0, ggml_nbytes(tensor_v));
                }
            }
        } else {
            fprintf(stderr, " warning %s not found\n", name.c_str());
        }
    }
    return true;
}

void ModelLoader::pre_process_tensors(std::string& name) {
    std::string key = "visual_projection";
    if (starts_with(name, key)) {
        name = "vision_model." + name;
    }
    key = "text_projection";
    if (starts_with(name, key)) {
        name = "text_model." + name;
    }
}

void ModelLoader::convert_tensor(void* src,
                                 ggml_type src_type,
                                 void* dst,
                                 ggml_type dst_type,
                                 int nrows,
                                 int n_per_row) {
    int n = nrows * n_per_row;
    if (src_type == dst_type) {
        size_t nbytes = n * ggml_type_size(src_type) / ggml_blck_size(src_type);
        memcpy(((char*)dst), ((char*)src), nbytes);
    } else if (src_type == GGML_TYPE_F32) {
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src, (ggml_fp16_t*)dst, n);
        } else {
            std::vector<float> imatrix(n_per_row, 1.0f);  // dummy importance matrix
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src, dst, 0, nrows, n_per_row, im);
        }
    } else if (dst_type == GGML_TYPE_F32) {
        if (src_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t*)src, (float*)dst, n);
        } else {
            auto qtype = ggml_internal_get_type_traits(src_type);
            if (qtype.to_float == NULL) {
                throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available",
                                                ggml_type_name(src_type)));
            }
            qtype.to_float(src, (float*)dst, n);
        }
    } else {
        auto qtype = ggml_internal_get_type_traits(src_type);
        if (qtype.to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available",
                                            ggml_type_name(src_type)));
        }
        std::vector<char> buf;
        buf.resize(sizeof(float) * n);
        char* src_data_f32 = buf.data();
        qtype.to_float(src, (float*)src_data_f32, n);
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src_data_f32, (ggml_fp16_t*)dst, n);
        } else {
            std::vector<float> imatrix(n_per_row, 1.0f);  // dummy importance matrix
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src_data_f32, dst, 0, nrows, n_per_row, im);
        }
    }
}