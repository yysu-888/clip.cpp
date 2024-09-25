#pragma once
#include <iostream>

enum CLIPVersion {
    OPENAI_CLIP_VIT_BASE_PATCH32 = 0,
    OPENAI_CLIP_VIT_LARGE_PATCH14,
    OPENAI_CLIP_VIT_large_patch14_336,
    OPENAI_CLIP_VIT_BASE_PATCH16,
    OFASYS_CHINESE_CLIP_VIT_HUGE_PATCH14,
    OFASYS_CHINESE_CLIP_VIT_LARGE_PATCH14_336,
    OFASYS_CHINESE_CLIP_VIT_BASE_PATCH16,
    OFASYS_CHINESE_CLIP_VIT_LARGE_PATCH14,
    VERSION_COUNT
};

const char* clip_version[] = {
    "openai_clip_vit_base_patch32",
    "openai_clip_vit_large_pathc14",
    "openai_clip_vit_large_patch14_336",
    "openai_clip_vit_base_patch16",
    "ofasys_chinese_clip_vit_huge_patch14",
    "ofasys_chinese_clip_vit_large_patch14_336",
    "ofasys_chinese_clip_vit_base_patch16",
    "ofasys_chinese_clip_vit_large_patch14"};

struct TextModelParam {
    TextModelParam(CLIPVersion version) {
        if (version == OPENAI_CLIP_VIT_BASE_PATCH32) {
            hidden_size             = 512;
            intermediate_size       = 2048;
            max_position_embeddings = 77;
            num_attention_heads     = 8;
            num_hidden_layers       = 12;
            vocab_size              = 49408;
            projection_dim          = 512;
        } else if (version == OPENAI_CLIP_VIT_LARGE_PATCH14) {
            hidden_size             = 768;
            intermediate_size       = 3072;
            max_position_embeddings = 77;
            num_attention_heads     = 12;
            num_hidden_layers       = 12;
            vocab_size              = 49408;
            projection_dim          = 768;
        } else if (version == OPENAI_CLIP_VIT_large_patch14_336) {
            hidden_size             = 768;
            intermediate_size       = 3072;
            max_position_embeddings = 77;
            num_attention_heads     = 12;
            num_hidden_layers       = 12;
            vocab_size              = 49408;
            projection_dim          = 768;
        } else if (version == OPENAI_CLIP_VIT_BASE_PATCH16) {
            hidden_size             = 512;
            intermediate_size       = 2048;
            max_position_embeddings = 77;
            num_attention_heads     = 8;
            num_hidden_layers       = 12;
            vocab_size              = 49408;
            projection_dim          = 512;
        } else if (version == OFASYS_CHINESE_CLIP_VIT_HUGE_PATCH14) {
            hidden_size             = 768;
            intermediate_size       = 3072;
            max_position_embeddings = 512;
            num_attention_heads     = 12;
            num_hidden_layers       = 12;
            vocab_size              = 21128;
            projection_dim          = 768;
            type_vocab_size         = 2;
        } else if (version == OFASYS_CHINESE_CLIP_VIT_LARGE_PATCH14_336) {
            hidden_size             = 768;
            intermediate_size       = 3072;
            max_position_embeddings = 512;
            num_attention_heads     = 12;
            num_hidden_layers       = 12;
            vocab_size              = 21128;
            projection_dim          = 768;
            type_vocab_size         = 2;
        } else if (version == OFASYS_CHINESE_CLIP_VIT_BASE_PATCH16) {
            hidden_size             = 768;
            intermediate_size       = 3072;
            max_position_embeddings = 512;
            num_attention_heads     = 12;
            num_hidden_layers       = 12;
            vocab_size              = 21128;
            projection_dim          = 512;
            type_vocab_size         = 2;
        } else if (version == OFASYS_CHINESE_CLIP_VIT_LARGE_PATCH14) {
            hidden_size             = 768;
            intermediate_size       = 3072;
            max_position_embeddings = 512;
            num_attention_heads     = 12;
            num_hidden_layers       = 12;
            vocab_size              = 21128;
            projection_dim          = 768;
            type_vocab_size         = 2;
        }
    }

public:
    int hidden_size;
    int intermediate_size;
    int max_position_embeddings;
    int num_attention_heads;
    int num_hidden_layers;
    int vocab_size;
    int projection_dim;
    int type_vocab_size;
};

typedef struct TextModelParam ClipTextModelParam;
struct VisionParam {
    VisionParam(CLIPVersion version) {
        if (version == OPENAI_CLIP_VIT_BASE_PATCH32) {
            hidden_size         = 768;
            image_size          = 224;
            intermediate_size   = 3072;
            num_attention_heads = 12;
            num_hidden_layers   = 12;
            patch_size          = 32;
            projection_dim      = 512;
        } else if (version == OPENAI_CLIP_VIT_LARGE_PATCH14) {
            hidden_size         = 1024;
            image_size          = 224;
            intermediate_size   = 4096;
            num_attention_heads = 16;
            num_hidden_layers   = 24;
            patch_size          = 14;
            projection_dim      = 768;
        } else if (version == OPENAI_CLIP_VIT_large_patch14_336) {
            hidden_size         = 1024;
            image_size          = 336;
            intermediate_size   = 4096;
            num_attention_heads = 16;
            num_hidden_layers   = 24;
            patch_size          = 14;
            projection_dim      = 768;
        } else if (version == OPENAI_CLIP_VIT_BASE_PATCH16) {
            hidden_size         = 768;
            image_size          = 224;
            intermediate_size   = 3072;
            num_attention_heads = 12;
            num_hidden_layers   = 12;
            patch_size          = 16;
            projection_dim      = 512;

        } else if (version == OFASYS_CHINESE_CLIP_VIT_HUGE_PATCH14) {
            hidden_size         = 1024;
            image_size          = 224;
            intermediate_size   = 4096;
            num_attention_heads = 16;
            num_hidden_layers   = 24;
            patch_size          = 14;
            projection_dim      = 768;
        } else if (version == OFASYS_CHINESE_CLIP_VIT_LARGE_PATCH14_336) {
            hidden_size         = 1024;
            image_size          = 336;
            intermediate_size   = 4096;
            num_attention_heads = 16;
            num_hidden_layers   = 24;
            patch_size          = 14;
            projection_dim      = 768;
        } else if (version == OFASYS_CHINESE_CLIP_VIT_BASE_PATCH16) {
            hidden_size         = 768;
            image_size          = 224;
            intermediate_size   = 3072;
            num_attention_heads = 12;
            num_hidden_layers   = 12;
            patch_size          = 16;
            projection_dim      = 512;
        } else if (version == OFASYS_CHINESE_CLIP_VIT_LARGE_PATCH14) {
            hidden_size         = 1024;
            image_size          = 224;
            intermediate_size   = 3072;
            num_attention_heads = 16;
            num_hidden_layers   = 24;
            patch_size          = 14;
            projection_dim      = 768;
        }
    }

public:
    int hidden_size;
    int image_size;
    int intermediate_size;
    int num_attention_heads;
    int num_hidden_layers;
    int patch_size;
    int projection_dim;
    float means[3] = {0.48145466, 0.4578275, 0.40821073};
    float stds[3]  = {0.26862954, 0.26130258, 0.27577711};
};
typedef struct VisionParam ClipVisionParam;
typedef struct ModelParam {
    CLIPVersion version = OPENAI_CLIP_VIT_BASE_PATCH32;
    ClipTextModelParam text_model_param;
    ClipVisionParam vision_model_param;
} ClipModelParam;