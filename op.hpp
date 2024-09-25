#pragma once
#include "common.hpp"

class Linear : public GGMLBlock {
protected:
    int64_t in_features;
    int64_t out_features;
    bool bias;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["weight"] =
            ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (bias) {
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_features);
        }
    }

public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true)
        : in_features(in_features), out_features(out_features), bias(bias) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        struct ggml_tensor* b = NULL;
        if (bias) {
            b = params["bias"];
        }
        return ggml_nn_linear(ctx, x, w, b);
    }
};

class LayerNorm : public GGMLBlock {
protected:
    int normalized_shape;
    float eps;
    bool elementwise_affine;
    bool bias;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        if (elementwise_affine) {
            params["weight"] =
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
            if (bias) {
                params["bias"] =
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
            }
        }
    }

public:
    LayerNorm(int normalized_shape, float eps = 1e-05f, bool elementwise_affine = true, bool bias = true)
        : normalized_shape(normalized_shape), eps(eps), elementwise_affine(elementwise_affine), bias(bias) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = NULL;
        struct ggml_tensor* b = NULL;

        if (elementwise_affine) {
            w = params["weight"];
            if (bias) {
                b = params["bias"];
            }
        }
        return ggml_nn_layer_norm(ctx, x, w, b, eps);
    }
};

class CLIPEmbeddings : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t vocab_size;
    int64_t num_positions;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["token_embedding.weight"] =
            ggml_new_tensor_2d(ctx, wtype, embed_dim, vocab_size);
        params["position_embedding.weight"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, num_positions);
    }

public:
    CLIPEmbeddings(int64_t embed_dim, int64_t vocab_size = 49408, int64_t num_positions = 77)
        : embed_dim(embed_dim), vocab_size(vocab_size), num_positions(num_positions) {}

    struct ggml_tensor* get_token_embed_weight() {
        return params["token_embedding.weight"];
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* input_ids,
                                struct ggml_tensor* custom_embed_weight = NULL) {
        // input_ids: [N, n_token]
        auto token_embed_weight    = params["token_embedding.weight"];
        auto position_embed_weight = params["position_embedding.weight"];

        GGML_ASSERT(input_ids->ne[0] == position_embed_weight->ne[1]);
        input_ids            = ggml_reshape_3d(ctx, input_ids, input_ids->ne[0], 1,
                                               input_ids->ne[1]);  // n x 1 x n_token
        auto token_embedding = ggml_get_rows(
            ctx,
            custom_embed_weight != NULL ? custom_embed_weight : token_embed_weight,
            input_ids);  //  ebd x vob

        token_embedding = ggml_reshape_3d(
            ctx, token_embedding, token_embedding->ne[0], token_embedding->ne[1],
            token_embedding->ne[3]);  //  ebd x n x1 x n_token

        // token_embedding + position_embedding
        auto x = ggml_add(ctx,  // ebd x n  x n_token ,ebd x n_p
                          token_embedding,
                          position_embed_weight);  // [N, n_token, embed_dim]

        return x;
    }
};

class VisionEmbeddings : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t num_channels;
    int64_t patch_size;
    int64_t image_size;
    int64_t num_patches;
    int64_t num_positions;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["patch_embedding.weight"] = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F16, patch_size, patch_size, num_channels, embed_dim);
        params["class_embedding"] =
            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, embed_dim);
        params["position_embedding.weight"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, num_positions);
    }

public:
    VisionEmbeddings(int64_t embed_dim, int64_t num_channels = 3, int64_t patch_size = 14, int64_t image_size = 224)
        : embed_dim(embed_dim), num_channels(num_channels), patch_size(patch_size), image_size(image_size) {
        num_patches   = (image_size / patch_size) * (image_size / patch_size);
        num_positions = num_patches + 1;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* pixel_values) {
        // pixel_values: [N, num_channels, image_size, image_size]
        // return: [N, num_positions, embed_dim]
        GGML_ASSERT(pixel_values->ne[0] == image_size &&
                    pixel_values->ne[1] == image_size &&
                    pixel_values->ne[2] == num_channels);

        auto patch_embed_weight    = params["patch_embedding.weight"];
        auto class_embed_weight    = params["class_embedding"];
        auto position_embed_weight = params["position_embedding.weight"];

        // concat(patch_embedding, class_embedding) + position_embedding
        struct ggml_tensor* patch_embedding;
        int64_t N = pixel_values->ne[3];
        patch_embedding =
            ggml_nn_conv_2d(ctx, pixel_values, patch_embed_weight, NULL, patch_size,
                            patch_size);  // [N, embed_dim, image_size // pacht_size,

        patch_embedding =
            ggml_reshape_3d(ctx, patch_embedding, num_patches, embed_dim,
                            N);  // [N, embed_dim, num_patches]
        patch_embedding =
            ggml_cont(ctx, ggml_permute(ctx, patch_embedding, 1, 0, 2,
                                        3));  // [N, num_patches, embed_dim]
        patch_embedding =
            ggml_reshape_4d(ctx, patch_embedding, 1, embed_dim, num_patches,
                            N);  // [N, num_patches, embed_dim, 1]

        struct ggml_tensor* class_embedding =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, N);
        class_embedding =
            ggml_repeat(ctx, class_embed_weight, class_embedding);  // [N, embed_dim]
        class_embedding = ggml_reshape_4d(ctx, class_embedding, 1, embed_dim, 1,
                                          N);  // [N, 1, embed_dim, 1]

        struct ggml_tensor* x = ggml_concat(ctx, class_embedding, patch_embedding, 2);  // [N, num_positions, embed_dim, 1]
        x                     = ggml_reshape_3d(ctx, x, embed_dim, num_positions,
                                                N);  // [N, num_positions, embed_dim]
        x                     = ggml_add(ctx, x, position_embed_weight);
        return x;  // [N, num_positions, embed_dim]
    }
};

class CLIPBertEmbeddings : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t vocab_size;
    int64_t num_positions;
    int64_t type_vocab_size;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["word_embeddings.weight"] =
            ggml_new_tensor_2d(ctx, wtype, embed_dim, vocab_size);
        params["position_embeddings.weight"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, num_positions);
        params["token_type_embeddings.weight"] =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, type_vocab_size);

        params["LayerNorm.weight"] =
            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, embed_dim);
        params["LayerNorm.bias"] =
            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, embed_dim);
    }

public:
    CLIPBertEmbeddings(int64_t embed_dim, int64_t vocab_size = 49408, int64_t num_positions = 77, int64_t type_vocab_size = 2)
        : embed_dim(embed_dim), vocab_size(vocab_size), num_positions(num_positions), type_vocab_size(type_vocab_size) {}

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* input_ids,
                                struct ggml_tensor* custom_embed_weight = NULL) {
        // input_ids: [N, n_token]
        auto token_embed_weight      = params["word_embeddings.weight"];
        auto position_embed_weight   = params["position_embeddings.weight"];
        auto token_type_embed_weight = params["token_type_embeddings.weight"];
        auto layer_norm_weight       = params["LayerNorm.weight"];
        auto layer_norm_bais         = params["LayerNorm.bias"];

        GGML_ASSERT(input_ids->ne[0] == position_embed_weight->ne[1]);
        input_ids            = ggml_reshape_3d(ctx, input_ids, input_ids->ne[0], 1,
                                               input_ids->ne[1]);  // n x 1 x n_token
        auto token_embedding = ggml_get_rows(ctx, token_embed_weight, input_ids);

        token_embedding = ggml_add(ctx, token_embedding, position_embed_weight);

        auto token_type_embed = ggml_repeat(ctx, ggml_view_2d(ctx, token_type_embed_weight, embed_dim, 1, embed_dim, 0), position_embed_weight);
        token_embedding       = ggml_add(ctx, token_type_embed, token_embedding);

        token_embedding = ggml_nn_layer_norm(ctx, token_embedding, layer_norm_weight, layer_norm_bais, EPS);
        return token_embedding;
    }
};
