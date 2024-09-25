#pragma once
#include <memory>
#include "bert_tokenizer.hpp"
#include "model_loader.hpp"
#include "op.hpp"
#include "tokenizer.hpp"

class CLIPMLP : public GGMLBlock {
public:
    CLIPMLP(int64_t d_model, int64_t intermediate_size) {
        blocks["fc1"] =
            std::shared_ptr<GGMLBlock>(new Linear(d_model, intermediate_size));
        blocks["fc2"] =
            std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, d_model));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, n_token, d_model]
        auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);

        x = fc1->forward(ctx, x);
        x = ggml_gelu_quick_inplace(ctx, x);
        x = fc2->forward(ctx, x);
        return x;
    }
};

class MultiheadAttention : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t n_head;
    std::string q_proj_name;
    std::string k_proj_name;
    std::string v_proj_name;
    std::string out_proj_name;

public:
    MultiheadAttention(int64_t embed_dim, int64_t n_head, bool qkv_proj_bias = true, bool out_proj_bias = true, std::string q_proj_name = "q_proj", std::string k_proj_name = "k_proj", std::string v_proj_name = "v_proj", std::string out_proj_name = "out_proj")
        : embed_dim(embed_dim), n_head(n_head), q_proj_name(q_proj_name), k_proj_name(k_proj_name), v_proj_name(v_proj_name), out_proj_name(out_proj_name) {
        blocks[q_proj_name] = std::shared_ptr<GGMLBlock>(
            new Linear(embed_dim, embed_dim, qkv_proj_bias));
        blocks[k_proj_name] = std::shared_ptr<GGMLBlock>(
            new Linear(embed_dim, embed_dim, qkv_proj_bias));
        blocks[v_proj_name] = std::shared_ptr<GGMLBlock>(
            new Linear(embed_dim, embed_dim, qkv_proj_bias));
        blocks[out_proj_name] = std::shared_ptr<GGMLBlock>(
            new Linear(embed_dim, embed_dim, out_proj_bias));
    }

    // x: [N, n_token, embed_dim]
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* masks = NULL, bool mask = false) {
        auto q_proj   = std::dynamic_pointer_cast<Linear>(blocks[q_proj_name]);
        auto k_proj   = std::dynamic_pointer_cast<Linear>(blocks[k_proj_name]);
        auto v_proj   = std::dynamic_pointer_cast<Linear>(blocks[v_proj_name]);
        auto out_proj = std::dynamic_pointer_cast<Linear>(blocks[out_proj_name]);

        struct ggml_tensor* q = q_proj->forward(ctx, x);
        struct ggml_tensor* k = k_proj->forward(ctx, x);
        struct ggml_tensor* v = v_proj->forward(ctx, x);

        x = ggml_nn_attention_ext(ctx, q, k, v, n_head, masks,
                                  mask);  // [N, n_token, embed_dim]

        x = out_proj->forward(ctx, x);  // [N, n_token, embed_dim]
        return x;
    }
};

struct CLIPChineseLayer : public GGMLBlock {
protected:
    int64_t d_model;  // hidden_size/embed_dim
    int64_t n_head;
    int64_t intermediate_size;

public:
    CLIPChineseLayer(int64_t d_model, int64_t n_head, int64_t intermediate_size)
        : d_model(d_model), n_head(n_head), intermediate_size(intermediate_size) {
        blocks["self_attn"] = std::shared_ptr<GGMLBlock>(new MultiheadAttention(d_model, n_head, true, true));
        blocks["layn_0"]    = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model));
        blocks["layn_1"]    = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model));
        blocks["linear_0"]  = std::shared_ptr<GGMLBlock>(new Linear(d_model, intermediate_size));
        blocks["linear_1"]  = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, d_model));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* padding_mask = NULL, bool mask = false) {
        // x: [N, n_token, d_model]
        auto self_attn   = std::dynamic_pointer_cast<MultiheadAttention>(blocks["self_attn"]);
        auto layer_norm0 = std::dynamic_pointer_cast<LayerNorm>(blocks["layn_0"]);
        auto layer_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["layn_1"]);
        auto linear_0    = std::dynamic_pointer_cast<Linear>(blocks["linear_0"]);
        auto linear_1    = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);

        x = layer_norm0->forward(ctx, ggml_add(ctx, x,
                                               self_attn->forward(ctx, x, padding_mask, mask)));
        x = ggml_add(ctx, x, linear_1->forward(ctx, ggml_gelu_quick(ctx, linear_0->forward(ctx, x))));
        x = layer_norm1->forward(ctx, x);
        return x;
    }
};

struct CLIPLayer : public GGMLBlock {
protected:
    int64_t d_model;  // hidden_size/embed_dim
    int64_t n_head;
    int64_t intermediate_size;

public:
    CLIPLayer(int64_t d_model, int64_t n_head, int64_t intermediate_size)
        : d_model(d_model), n_head(n_head), intermediate_size(intermediate_size) {
        blocks["self_attn"] = std::shared_ptr<GGMLBlock>(new MultiheadAttention(d_model, n_head, true, true));

        blocks["layer_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model));
        blocks["layer_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model));

        blocks["mlp"] =
            std::shared_ptr<GGMLBlock>(new CLIPMLP(d_model, intermediate_size));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* padding_mask = NULL, bool mask = false) {
        // x: [N, n_token, d_model]
        auto self_attn =
            std::dynamic_pointer_cast<MultiheadAttention>(blocks["self_attn"]);
        auto layer_norm1 =
            std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm1"]);
        auto layer_norm2 =
            std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm2"]);
        auto mlp = std::dynamic_pointer_cast<CLIPMLP>(blocks["mlp"]);

        x = ggml_add(ctx, x,
                     self_attn->forward(ctx, layer_norm1->forward(ctx, x), padding_mask, mask));
        x = ggml_add(ctx, x, mlp->forward(ctx, layer_norm2->forward(ctx, x)));
        return x;
    }
};

class CLIPEncoder : public GGMLBlock {
protected:
    int64_t n_layer;

public:
    CLIPEncoder(int64_t n_layer, int64_t d_model, int64_t n_head, int64_t intermediate_size)
        : n_layer(n_layer) {
        for (int i = 0; i < n_layer; i++) {
            std::string name = "layers." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(
                new CLIPLayer(d_model, n_head, intermediate_size));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* padding_mask = NULL, int clip_skip = -1, bool mask = true) {
        // x: [N, n_token, d_model] int clip_skip = -1, bool mask = true) {
        // x: [N, n_token, d_model]
        int layer_idx = n_layer - 1;
        if (clip_skip > 0) {
            layer_idx = n_layer - clip_skip;
        }

        for (int i = 0; i < n_layer; i++) {
            std::string name = "layers." + std::to_string(i);
            auto layer       = std::dynamic_pointer_cast<CLIPLayer>(blocks[name]);
            x                = layer->forward(ctx, x, padding_mask, mask);  // [N, n_token, d_model]
        }
        return x;
    }
};

class CLIPChineseEncoder : public GGMLBlock {
protected:
    int64_t n_layer;

public:
    CLIPChineseEncoder(int64_t n_layer, int64_t d_model, int64_t n_head, int64_t intermediate_size)
        : n_layer(n_layer) {
        for (int i = 0; i < n_layer; i++) {
            std::string name = "layer." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(
                new CLIPChineseLayer(d_model, n_head, intermediate_size));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* padding_mask, int clip_skip = -1, bool mask = true) {
        // x: [N, n_token, d_model]
        int layer_idx = n_layer - 1;
        if (clip_skip > 0) {
            layer_idx = n_layer - clip_skip;
        }

        for (int i = 0; i < n_layer; i++) {
            std::string name = "layer." + std::to_string(i);
            auto layer       = std::dynamic_pointer_cast<CLIPChineseLayer>(blocks[name]);
            x                = layer->forward(ctx, x, padding_mask, mask);  // [N, n_token, d_model]
        }
        return x;
    }
};

class ClipChineseTextModel : public GGMLBlock {
public:
    ClipTextModelParam param;
    CLIPVersion version;
    ClipChineseTextModel(ClipTextModelParam text_model_param, CLIPVersion version)
        : param(text_model_param), version(version) {
        blocks["embeddings"]      = std::shared_ptr<GGMLBlock>(new CLIPBertEmbeddings(text_model_param.hidden_size, text_model_param.vocab_size, text_model_param.max_position_embeddings));
        blocks["encoder"]         = std::shared_ptr<GGMLBlock>(new CLIPChineseEncoder(text_model_param.num_hidden_layers, text_model_param.hidden_size, text_model_param.num_attention_heads, text_model_param.intermediate_size));
        blocks["text_projection"] = std::shared_ptr<GGMLBlock>(new Linear(text_model_param.hidden_size, text_model_param.projection_dim, false));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input_ids, float idx, struct ggml_tensor* padding_mask, struct ggml_tensor* tkn_embeddings = NULL, size_t max_token_idx = 77, bool mask = false, bool return_pooled = true, bool normalize = true) {
        auto embeddings      = std::dynamic_pointer_cast<CLIPBertEmbeddings>(blocks["embeddings"]);
        auto encoder         = std::dynamic_pointer_cast<CLIPChineseEncoder>(blocks["encoder"]);
        auto text_projection = std::dynamic_pointer_cast<Linear>(blocks["text_projection"]);

        struct ggml_tensor* x = embeddings->forward(ctx, input_ids, tkn_embeddings);
        x                     = encoder->forward(ctx, x, padding_mask, -1, mask);

        if (return_pooled) {
            struct ggml_tensor* pooled = ggml_cont(ctx, ggml_view_1d(ctx, x, param.hidden_size, x->nb[1] * 0));
            x                          = text_projection->forward(ctx, pooled);  // [N, n_token, d_model]
        }
        if (normalize) {
            ggml_tensor* length = ggml_sqrt(ctx, ggml_mul_mat(ctx, x, x));
            x                   = ggml_div_inplace(ctx, x, length);
        }
        return x;  // [N, n_token, hidden_size]
    }
};

class ClipTextModel : public GGMLBlock {
public:
    ClipTextModelParam param;
    CLIPVersion version;
    ClipTextModel(ClipTextModelParam text_model_param, CLIPVersion version = OPENAI_CLIP_VIT_BASE_PATCH32)
        : param(text_model_param), version(version) {
        blocks["embeddings"]       = std::shared_ptr<GGMLBlock>(new CLIPEmbeddings(text_model_param.hidden_size, text_model_param.vocab_size, text_model_param.max_position_embeddings));
        blocks["encoder"]          = std::shared_ptr<GGMLBlock>(new CLIPEncoder(text_model_param.num_hidden_layers, text_model_param.hidden_size, text_model_param.num_attention_heads, text_model_param.intermediate_size));
        blocks["final_layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(text_model_param.hidden_size, 1e-05f, true, true));
        blocks["text_projection"]  = std::shared_ptr<GGMLBlock>(new Linear(text_model_param.hidden_size, text_model_param.projection_dim, false));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input_ids, float idx, struct ggml_tensor* padding_mask, struct ggml_tensor* tkn_embeddings = NULL, size_t max_token_idx = 77, bool mask = true, bool return_pooled = true, bool normalize = true) {
        auto embeddings       = std::dynamic_pointer_cast<CLIPEmbeddings>(blocks["embeddings"]);
        auto encoder          = std::dynamic_pointer_cast<CLIPEncoder>(blocks["encoder"]);
        auto final_layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["final_layer_norm"]);
        auto text_projection  = std::dynamic_pointer_cast<Linear>(blocks["text_projection"]);

        struct ggml_tensor* x = embeddings->forward(ctx, input_ids, tkn_embeddings);
        x                     = encoder->forward(ctx, x, padding_mask, -1, mask);
        x                     = final_layer_norm->forward(ctx, x);  // [N, n_token, d_model]
        if (return_pooled) {
            struct ggml_tensor* pooled = ggml_cont(ctx, ggml_view_1d(ctx, x, param.hidden_size, x->nb[1] * (idx)));
            x                          = text_projection->forward(ctx, pooled);  // [N, n_token, d_model]
        }
        if (normalize) {
            ggml_tensor* length = ggml_sqrt(ctx, ggml_mul_mat(ctx, x, x));
            x                   = ggml_div_inplace(ctx, x, length);
        }
        return x;  // [N, n_token, hidden_size]
    }
};

class ClipVisionModel : public GGMLBlock {
public:
    ClipVisionParam param;
    CLIPVersion version;
    ClipVisionModel(ClipVisionParam vision_model_param, CLIPVersion version = OPENAI_CLIP_VIT_BASE_PATCH32)
        : param(vision_model_param), version(version) {
        blocks["embeddings"]        = std::shared_ptr<GGMLBlock>(new VisionEmbeddings(vision_model_param.hidden_size, 3, vision_model_param.patch_size, vision_model_param.image_size));
        blocks["pre_layrnorm"]      = std::shared_ptr<GGMLBlock>(new LayerNorm(vision_model_param.hidden_size, 1e-05f, true, true));
        blocks["encoder"]           = std::shared_ptr<GGMLBlock>(new CLIPEncoder(vision_model_param.num_hidden_layers, vision_model_param.hidden_size, vision_model_param.num_attention_heads, vision_model_param.intermediate_size));
        blocks["post_layernorm"]    = std::shared_ptr<GGMLBlock>(new LayerNorm(vision_model_param.hidden_size, 1e-05f, true, true));
        blocks["visual_projection"] = std::shared_ptr<GGMLBlock>(new Linear(vision_model_param.hidden_size, vision_model_param.projection_dim, false));
    }
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* pixel_values, bool return_pooled = true, bool normalize = true) {
        auto embeddings        = std::dynamic_pointer_cast<VisionEmbeddings>(blocks["embeddings"]);
        auto pre_layernorm     = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_layrnorm"]);
        auto encoder           = std::dynamic_pointer_cast<CLIPEncoder>(blocks["encoder"]);
        auto post_layernorm    = std::dynamic_pointer_cast<LayerNorm>(blocks["post_layernorm"]);
        auto visual_projection = std::dynamic_pointer_cast<Linear>(blocks["visual_projection"]);

        auto x = embeddings->forward(ctx, pixel_values);
        x      = pre_layernorm->forward(ctx, x);
        x      = encoder->forward(ctx, x, NULL, -1, false);

        struct ggml_tensor* pooled = ggml_cont(ctx, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0));
        x                          = post_layernorm->forward(ctx, pooled);

        if (return_pooled) {
            x = visual_projection->forward(ctx, x);
        }

        if (normalize) {
            ggml_tensor* length = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, x, x)));  // x ->[b,n_token,dim],b=1
            x                   = ggml_div_inplace(ctx, x, length);
        }
        return x;
    }
};

template <class VisionModel>
struct CLIPVisionModelRunner : public GGMLRunner {
    std::unique_ptr<VisionModel> model;
    CLIPVersion version;

    CLIPVisionModelRunner(ggml_backend_t backend,
                          ggml_type wtype,
                          ClipVisionParam param,
                          CLIPVersion version = OPENAI_CLIP_VIT_BASE_PATCH32)
        : GGMLRunner(backend, wtype) {
        model = std::make_unique<VisionModel>(param, version);
        model->init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "clip_vision";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix = "vision_model") {
        model->get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* pixel_values) {
        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        pixel_values = to_backend(pixel_values);

        struct ggml_tensor* out = model->forward(compute_ctx, pixel_values);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    void compute(const int n_threads,
                 struct ggml_tensor* input_pixel,
                 ggml_tensor** output,
                 ggml_context* output_ctx = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(input_pixel);
        };
        GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
    }

    void infer(image_u8_t images_u8, std::vector<float>& vec_out) {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(2 * 1024 * 1024);  // 20 MB
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        image_f32_t image_f32 = image_preprocess(images_u8, model->param.image_size, model->param.means, model->param.stds);

        struct ggml_context* work_ctx = ggml_init(params);
        GGML_ASSERT(work_ctx != NULL);
        {
            auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, image_f32.width, image_f32.height, 3, 1);
            image_f32_to_tensor(image_f32.data, x, false);

            struct ggml_tensor* out = NULL;

            float t0 = ggml_time_ms();
            compute(2, x, &out, work_ctx);
            float t1 = ggml_time_ms();
            printf("%s consuming time=%5.f ms\n", "clip_vision_modelrunner", t1 - t0);

            vec_out.resize(ggml_nelements(out));
            memcpy(vec_out.data(), ggml_get_data_f32(out), ggml_nbytes(out));
#ifdef CLIP_DEBUG
            print_ggml_tensor(out);
#endif
        }
        free(image_f32.data);
        ggml_free(work_ctx);
    }
};

template <class TextModel>
struct CLIPTextModelRunner : public GGMLRunner {
    std::unique_ptr<TextModel> model;
    CLIPVersion version;

    CLIPTextModelRunner(ggml_backend_t backend,
                        ggml_type wtype,
                        ClipTextModelParam param,
                        CLIPVersion version = OPENAI_CLIP_VIT_BASE_PATCH32)
        : GGMLRunner(backend, wtype) {
        model = std::make_unique<TextModel>(param, version);
        model->init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "clip_text";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix = "text_model") {
        model->get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* input_ids, float idx, struct ggml_tensor* padding_mask) {
        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        input_ids    = to_backend(input_ids);
        padding_mask = to_backend(padding_mask);

        struct ggml_tensor* out = model->forward(compute_ctx, input_ids, idx, padding_mask);
        ggml_build_forward_expand(gf, out);

        return gf;
    }

    void compute(const int n_threads,
                 struct ggml_tensor* input_ids,
                 float idx,
                 struct ggml_tensor* padding_mask,
                 ggml_tensor** output,
                 ggml_context* output_ctx = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(input_ids, idx, padding_mask);
        };
        GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
    }

    void infer(std::vector<int>& ids_vec, std::vector<float>& vec_out) {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(20 * 1024 * 1024);  // 20 MB
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        struct ggml_context* work_ctx = ggml_init(params);
        GGML_ASSERT(work_ctx != NULL);

        {
            int idx                             = get_argmax(ids_vec);
            std::vector<float> padding_mask_vec = get_padding_mask(ids_vec);
            auto ids                            = vector_to_ggml_tensor_int(work_ctx, ids_vec);
            struct ggml_tensor* padding_mask    = vector_to_ggml_tensor(work_ctx, padding_mask_vec);

            struct ggml_tensor* out = NULL;

            float t0 = ggml_time_ms();
            compute(2, ids, idx, padding_mask, &out, work_ctx);
            float t1 = ggml_time_ms();
            printf("%s consuming time=%5.f ms\n", "clip_text_modelrunner", t1 - t0);
            vec_out.resize(ggml_nelements(out));
            memcpy(vec_out.data(), ggml_get_data_f32(out), ggml_nbytes(out));
#ifdef CLIP_DEBUG
            print_ggml_tensor(out);
#endif
        }
        ggml_free(work_ctx);
    }
};

template <class VisionModel, class TextModel>
class ClipModel {
public:
    ClipModel(ggml_backend_t backend, ggml_type wtype, ClipTextModelParam text_param, ClipVisionParam vision_param, CLIPVersion version = OPENAI_CLIP_VIT_BASE_PATCH32, bool enable_vision = true, bool enable_text = true)
        : enable_vision(enable_vision), enable_text(enable_text), version(version), backend(backend) {
        if (enable_vision) {
            vision_runner = std::make_unique<CLIPVisionModelRunner<VisionModel>>(backend, wtype, vision_param, version);
            vision_runner->alloc_params_buffer();
        }
        if (enable_text) {
            text_runner = std::make_unique<CLIPTextModelRunner<TextModel>>(backend, wtype, text_param, version);
            text_runner->alloc_params_buffer();
        }
        if ((!enable_vision) && (!enable_text)) {
            std::runtime_error("vision model and text model can not disable");
        }
    }
    void load_weight(std::string model_path) {
        tensors.clear();
        if (enable_vision) {
            vision_runner->alloc_params_buffer();
            vision_runner->get_param_tensors(tensors);
        }
        if (enable_text) {
            text_runner->alloc_params_buffer();
            text_runner->get_param_tensors(tensors);
        }
        if (enable_vision || enable_vision) {
            model_loader = std::make_unique<ModelLoader>(version);
            model_loader->init_gguf_file(model_path);
            model_loader->load_tensor(tensors, backend);
        }
    }

    void get_image_features(image_u8_t images_u8, std::vector<float>& vis_vec_out) {
        if (enable_vision) {
            vision_runner->infer(images_u8, vis_vec_out);
        } else {
            std::runtime_error("model disable vision embeding");
        }
    }
    void get_text_features(std::string label, std::vector<float>& text_vec_out) {
        if (enable_text) {
            size_t max_token_id = text_runner->model->param.max_position_embeddings;
            std::vector<int> ids;
            if (version == OPENAI_CLIP_VIT_BASE_PATCH32 || version == OPENAI_CLIP_VIT_LARGE_PATCH14 ||
                version == OPENAI_CLIP_VIT_BASE_PATCH16 || version == OPENAI_CLIP_VIT_large_patch14_336) {
                ids = tokenizer.tokenize(label, nullptr, max_token_id, true);
            } else if (version == OFASYS_CHINESE_CLIP_VIT_HUGE_PATCH14 || version == OFASYS_CHINESE_CLIP_VIT_LARGE_PATCH14_336 ||
                       version == OFASYS_CHINESE_CLIP_VIT_BASE_PATCH16 || version == OFASYS_CHINESE_CLIP_VIT_LARGE_PATCH14) {
                ids = bert_tokenizer.encode(label, max_token_id, true, true);
            }
            text_runner->infer(ids, text_vec_out);
        } else {
            std::runtime_error("model disable text embeding");
        }
    }

private:
    std::unique_ptr<CLIPVisionModelRunner<VisionModel>> vision_runner;
    std::unique_ptr<CLIPTextModelRunner<TextModel>> text_runner;
    std::unique_ptr<ModelLoader> model_loader;
    bool enable_vision;
    bool enable_text;
    CLIPVersion version;
    ggml_backend_t backend;
    CLIPTokenizer tokenizer;
    BertTokenizer bert_tokenizer;
    std::map<std::string, struct ggml_tensor*> tensors;
};
