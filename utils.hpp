#pragma once
#include <iostream>
#include <sstream>
#include <string>

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <codecvt>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "vocab.hpp"

#define EPS 1e-05

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    float* data;
} image_f32_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} image_u8_t;

const char* sd_get_system_info() {
    static char buffer[1024];
    std::stringstream ss;
    ss << "System Info: \n";
    ss << "    BLAS = " << ggml_cpu_has_blas() << std::endl;
    ss << "    SSE3 = " << ggml_cpu_has_sse3() << std::endl;
    ss << "    AVX = " << ggml_cpu_has_avx() << std::endl;
    ss << "    AVX2 = " << ggml_cpu_has_avx2() << std::endl;
    ss << "    AVX512 = " << ggml_cpu_has_avx512() << std::endl;
    ss << "    AVX512_VBMI = " << ggml_cpu_has_avx512_vbmi() << std::endl;
    ss << "    AVX512_VNNI = " << ggml_cpu_has_avx512_vnni() << std::endl;
    ss << "    FMA = " << ggml_cpu_has_fma() << std::endl;
    ss << "    NEON = " << ggml_cpu_has_neon() << std::endl;
    ss << "    ARM_FMA = " << ggml_cpu_has_arm_fma() << std::endl;
    ss << "    F16C = " << ggml_cpu_has_f16c() << std::endl;
    ss << "    FP16_VA = " << ggml_cpu_has_fp16_va() << std::endl;
    ss << "    WASM_SIMD = " << ggml_cpu_has_wasm_simd() << std::endl;
    ss << "    VSX = " << ggml_cpu_has_vsx() << std::endl;
    snprintf(buffer, sizeof(buffer), "%s", ss.str().c_str());
    return buffer;
}

std::string format(const char* fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string load_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(merges_utf8_c_str),
                                sizeof(merges_utf8_c_str));
    return merges_utf8_str;
}

static bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

bool starts_with(const std::string& str, const std::string& start) {
    if (str.find(start) == 0) {
        return true;
    }
    return false;
}

bool contains(const std::string& str, const std::string& substr) {
    if (str.find(substr) != std::string::npos) {
        return true;
    }
    return false;
}

std::string ltrim(const std::string& s) {
    auto it = std::find_if(s.begin(), s.end(),
                           [](int ch) { return !std::isspace(ch); });
    return std::string(it, s.end());
}

std::string rtrim(const std::string& s) {
    auto it = std::find_if(s.rbegin(), s.rend(),
                           [](int ch) { return !std::isspace(ch); });
    return std::string(s.begin(), it.base());
}

std::string trim(const std::string& s) {
    return rtrim(ltrim(s));
}

std::u32string utf8_to_utf32(const std::string& utf8_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.from_bytes(utf8_str);
}

std::string utf32_to_utf8(const std::u32string& utf32_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.to_bytes(utf32_str);
}

std::u32string unicode_value_to_utf32(int unicode_value) {
    std::u32string utf32_string = {static_cast<char32_t>(unicode_value)};
    return utf32_string;
}

std::vector<std::pair<int, std::u32string>> bytes_to_unicode() {
    std::vector<std::pair<int, std::u32string>> byte_unicode_pairs;
    std::set<int> byte_set;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(
            std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 161; b <= 172; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(
            std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 174; b <= 255; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(
            std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_set.find(b) == byte_set.end()) {
            byte_unicode_pairs.push_back(
                std::pair<int, std::u32string>(b, unicode_value_to_utf32(n + 256)));
            ++n;
        }
    }
    return byte_unicode_pairs;
}

/// /////////////////////
static inline struct ggml_tensor* ggml_nn_linear(struct ggml_context* ctx,
                                                 struct ggml_tensor* x,
                                                 struct ggml_tensor* w,
                                                 struct ggml_tensor* b) {
    x = ggml_mul_mat(ctx, w, x);
    if (b != NULL) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}

static inline struct ggml_tensor*
ggml_nn_attention_ext(struct ggml_context* ctx, struct ggml_tensor* q, struct ggml_tensor* k, struct ggml_tensor* v, int64_t n_head, struct ggml_tensor* mask = NULL, bool diag_mask_inf = false) {
    int64_t L_q = q->ne[1];
    int64_t L_k = k->ne[1];
    int64_t C   = q->ne[0];
    int64_t N   = q->ne[2];

    int64_t d_head = C / n_head;
    float scale    = (1.0f / sqrt((float)d_head));

    q = ggml_reshape_4d(ctx, q, d_head, n_head, L_q,
                        N);  // [N, L_q, n_head, d_head]
    q = ggml_cont(ctx,
                  ggml_permute(ctx, q, 0, 2, 1, 3));  // [N, n_head, L_q, d_head]
    q = ggml_reshape_3d(ctx, q, d_head, L_q,
                        n_head * N);  // [N * n_head, L_q, d_head]

    k = ggml_reshape_4d(ctx, k, d_head, n_head, L_k,
                        N);  // [N, L_k, n_head, d_head]
    k = ggml_cont(ctx,
                  ggml_permute(ctx, k, 0, 2, 1, 3));  // [N, n_head, L_k, d_head]
    k = ggml_reshape_3d(ctx, k, d_head, L_k,
                        n_head * N);  // [N * n_head, L_k, d_head]

    v = ggml_reshape_4d(ctx, v, d_head, n_head, L_k,
                        N);  // [N, L_k, n_head, d_head]
    v = ggml_cont(ctx,
                  ggml_permute(ctx, v, 1, 2, 0, 3));  // [N, n_head, d_head, L_k]
    v = ggml_reshape_3d(ctx, v, L_k, d_head,
                        n_head * N);  // [N * n_head, d_head, L_k]

    auto kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, L_q, L_k]
    kq      = ggml_scale_inplace(ctx, kq, scale);
    if (mask) {
        mask = ggml_repeat(ctx, mask, kq);
        kq   = ggml_add(ctx, kq, mask);
    }
    if (diag_mask_inf) {
        kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
    }
    kq = ggml_soft_max_inplace(ctx, kq);

    auto kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, L_q, d_head]

    kqv = ggml_reshape_4d(ctx, kqv, d_head, L_q, n_head,
                          N);  // [N, n_head, L_q, d_head]
    kqv = ggml_cont(
        ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));              // [N, L_q, n_head, d_head]
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);  // [N, L_q, C]

    return kqv;
}

static inline ggml_tensor* ggml_nn_layer_norm(struct ggml_context* ctx,
                                              struct ggml_tensor* x,
                                              struct ggml_tensor* w,
                                              struct ggml_tensor* b,
                                              float eps = EPS) {
    x = ggml_norm(ctx, x, eps);
    if (w != NULL) {
        x = ggml_mul(ctx, x, w);
        if (b != NULL) {
            x = ggml_add(ctx, x, b);
        }
    }
    return x;
}

static inline struct ggml_tensor*
ggml_nn_conv_2d(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b, int s0 = 1, int s1 = 1, int p0 = 0, int p1 = 0, int d0 = 1, int d1 = 1) {
    x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
    if (b != NULL) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        // b = ggml_repeat(ctx, b, x);
        x = ggml_add(ctx, x, b);
    }
    return x;
}

static inline ggml_fp16_t ggml_tensor_get_f16(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
    return *(ggml_fp16_t*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

static inline int ggml_tensor_get_i32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    if (tensor->buffer != NULL) {
        float value;
        ggml_backend_tensor_get(tensor, &value, i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0], sizeof(int));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(int));
    return *(int*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

static inline float ggml_tensor_get_f32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    if (tensor->buffer != NULL) {
        float value;
        ggml_backend_tensor_get(tensor, &value, i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0], sizeof(float));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

static inline void print_ggml_tensor(struct ggml_tensor* tensor, bool shape_only = false, const char* mark = "") {
    printf("%s (%s): shape(%d, %d, %d, %d)\n", mark, ggml_type_name(tensor->type), int(tensor->ne[0]), int(tensor->ne[1]), int(tensor->ne[2]), int(tensor->ne[3]));
    fflush(stdout);
    if (shape_only) {
        return;
    }
    int range = 5;
    for (int i = 0; i < tensor->ne[3]; i++) {
        if (i >= range && i + range < tensor->ne[3]) {
            continue;
        }
        for (int j = 0; j < tensor->ne[2]; j++) {
            if (j >= range && j + range < tensor->ne[2]) {
                continue;
            }
            for (int k = 0; k < tensor->ne[1]; k++) {
                if (k >= range && k + range < tensor->ne[1]) {
                    continue;
                }
                for (int l = 0; l < tensor->ne[0]; l++) {
                    if (l >= range && l + range < tensor->ne[0]) {
                        continue;
                    }
                    if (tensor->type == GGML_TYPE_F32) {
                        printf("  [%d, %d, %d, %d] = %f\n", i, j, k, l, ggml_tensor_get_f32(tensor, l, k, j, i));
                    } else if (tensor->type == GGML_TYPE_F16) {
                        printf("  [%d, %d, %d, %d] = %i\n", i, j, k, l, ggml_tensor_get_f16(tensor, l, k, j, i));
                    } else if (tensor->type == GGML_TYPE_I32) {
                        printf("  [%d, %d, %d, %d] = %i\n", i, j, k, l, ggml_tensor_get_i32(tensor, l, k, j, i));
                    }
                    fflush(stdout);
                }
            }
        }
    }
}

static inline struct ggml_tensor* vector_to_ggml_tensor(struct ggml_context* ctx,
                                                        const std::vector<float>& vec) {
    struct ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

static inline size_t ggml_tensor_num(ggml_context* ctx) {
    size_t num = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr;
         t              = ggml_get_next_tensor(ctx, t)) {
        num++;
    }
    return num;
}

static inline void ggml_tensor_set_f32(struct ggml_tensor* tensor, float value, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]) = value;
}

void image_f32_to_tensor(const float* image_data,
                         struct ggml_tensor* output,
                         bool scale = true) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    GGML_ASSERT(channels == 3 && output->type == GGML_TYPE_F32);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value = *((float*)image_data + iy * width * channels + ix * channels + k);
                if (scale) {
                    value /= 255.f;
                }
                ggml_tensor_set_f32(output, value, ix, iy, k);
            }
        }
    }
}

static inline void
ggml_backend_tensor_get_and_sync(ggml_backend_t backend,
                                 const struct ggml_tensor* tensor,
                                 void* data,
                                 size_t offset,
                                 size_t size) {
#if defined(GGML_USE_CUBLAS)
    if (!ggml_backend_is_cpu(backend)) {
        ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
        ggml_backend_synchronize(backend);
    } else {
        ggml_backend_tensor_get(tensor, data, offset, size);
    }
#else
    ggml_backend_tensor_get(tensor, data, offset, size);
#endif
}

image_u8_t image_f32_t_to_image_8u_t(image_f32_t image) {
    image_u8_t converted_image;
    converted_image.width   = image.width;
    converted_image.height  = image.height;
    converted_image.channel = image.channel;

    converted_image.data = (uint8_t*)malloc(image.width * image.height * image.channel * sizeof(uint8_t));

    for (int i = 0; i < image.width * image.height * image.channel; i++) {
        converted_image.data[i] = (uint8_t)(image.data[i] * 255);
    }

    return converted_image;
}

image_f32_t image_8u_t_to_image_f32_t(image_u8_t image) {
    image_f32_t converted_image;
    converted_image.width   = image.width;
    converted_image.height  = image.height;
    converted_image.channel = image.channel;

    converted_image.data = (float*)malloc(image.width * image.height * image.channel * sizeof(float));

    for (int i = 0; i < image.width * image.height * image.channel; i++) {
        converted_image.data[i] = (float)image.data[i];
    }

    return converted_image;
}

static inline struct ggml_tensor* vector_to_ggml_tensor_int_2d(struct ggml_context* ctx,
                                                               std::vector<std::vector<int>>& vec) {
    struct ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, vec[0].size(), vec.size());
    int rows              = t->nb[1];
    for (int i = 0; i < vec.size(); i++) {
        memcpy((float*)t->data + i * rows, (const void*)vec[i].data(), sizeof(float) * vec[i].size());
    }
    return t;
}

static inline struct ggml_tensor* vector_to_ggml_tensor_int(struct ggml_context* ctx,
                                                            std::vector<int>& vec) {
    struct ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

static inline void image_to_tensor(const float* image_data,
                                   struct ggml_tensor* output,
                                   bool scale = true) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    GGML_ASSERT(channels == 3 && output->type == GGML_TYPE_F32);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value = *(image_data + iy * width * channels + ix * channels + k);
                if (scale) {
                    value /= 255.f;
                }
                ggml_tensor_set_f32(output, value, ix, iy, k);
            }
        }
    }
}

float interpolate(float v1, float v2, float v3, float v4, float x_ratio, float y_ratio) {
    return v1 * (1 - x_ratio) * (1 - y_ratio) + v2 * x_ratio * (1 - y_ratio) + v3 * (1 - x_ratio) * y_ratio + v4 * x_ratio * y_ratio;
}

image_f32_t image_preprocess(image_u8_t image_u8, int size, float* means, float* stds) {
    float scale = (float)size / fmin(image_u8.width, image_u8.height);

    image_f32_t image = image_8u_t_to_image_f32_t(image_u8);

    int new_width       = (int)(scale * image.width);
    int new_height      = (int)(scale * image.height);
    image.channel       = 3;
    float* resized_data = (float*)malloc(new_width * new_height * image.channel * sizeof(float));

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            float original_x = (float)x * image.width / new_width;
            float original_y = (float)y * image.height / new_height;

            int x1 = (int)original_x;
            int y1 = (int)original_y;
            int x2 = x1 + 1;
            int y2 = y1 + 1;

            for (int k = 0; k < image.channel; k++) {
                float v1 = *(image.data + y1 * image.width * image.channel + x1 * image.channel + k);
                float v2 = *(image.data + y1 * image.width * image.channel + x2 * image.channel + k);
                float v3 = *(image.data + y2 * image.width * image.channel + x1 * image.channel + k);
                float v4 = *(image.data + y2 * image.width * image.channel + x2 * image.channel + k);

                float x_ratio = original_x - x1;
                float y_ratio = original_y - y1;

                float value = interpolate(v1, v2, v3, v4, x_ratio, y_ratio);

                *(resized_data + y * new_width * image.channel + x * image.channel + k) = value;
            }
        }
    }

    int h = (new_height - size) / 2;
    int w = (new_width - size) / 2;

    image_f32_t result;
    result.width   = size;
    result.height  = size;
    result.channel = image.channel;
    result.data    = (float*)malloc(size * size * image.channel * sizeof(float));

    for (int k = 0; k < image.channel; k++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                *(result.data + i * size * image.channel + j * image.channel + k) =
                    fmin(fmax(*(resized_data + (i + h) * new_width * image.channel + (j + w) * image.channel + k), 0.0f), 255.0f) / 255.0f;
            }
        }
    }

    free(resized_data);

    for (int k = 0; k < image.channel; k++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // *(result.data + i * size * image.channel + j * image.channel + k) = 0.5f;
                int offset  = i * size * image.channel + j * image.channel + k;
                float value = *(result.data + offset);
                value       = (value - means[k]) / stds[k];
                // value = 0.5f;
                *(result.data + offset) = value;
            }
        }
    }

    return result;
}

int get_argmax(std::vector<int>& inp) {
    int max_idx   = 0;
    int max_value = -1;
    for (int i = 0; i < inp.size(); i++) {
        if (inp[i] > max_value) {
            max_value = inp[i];
            max_idx   = i;
        }
    }
    return max_idx;
}

std::vector<float> get_padding_mask(std::vector<int>& inp) {
    std::vector<float> padding_mask;
    for (int i = 0; i < inp.size(); i++) {
        if (inp[i] != 0) {
            padding_mask.push_back(0.f);
        } else {
            padding_mask.push_back(-1e5);
        }
    }
    return padding_mask;
}

std::vector<int> get_argmax_vec(std::vector<std::vector<int>>& inp) {
    std::vector<int> max_index;
    for (int i = 0; i < inp.size(); i++) {
        int max_idx = get_argmax(inp[i]);
        max_index.push_back(max_idx);
    }
    return max_index;
}

bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

class VectorIO {
public:
    VectorIO() = default;
    void vec_store(std::string file_path, std::vector<std::vector<float>>& vec);
    void vec_load(std::string file_path, std::vector<std::vector<float>>& vec);
    ~VectorIO() = default;
};

void VectorIO::vec_store(std::string file_path, std::vector<std::vector<float>>& vec) {
    std::ofstream file(file_path, std::ios::out | std::ios::binary);
    if (file.is_open()) {
        size_t rows = vec.size();
        size_t cols = vec[0].size();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));

        for (const auto& v : vec) {
            file.write(reinterpret_cast<const char*>(v.data()), cols * sizeof(float));
        }
        file.close();
    } else {
        fprintf(stderr, "failed to open the file for writing");
    }
}

void VectorIO::vec_load(std::string file_path, std::vector<std::vector<float>>& vec) {
    std::ifstream file(file_path, std::ios::in | std::ios::binary);
    if (file.is_open()) {
        size_t rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

        vec.resize(rows);
        for (int k = 0; k < rows; ++k) {
            vec[k].resize(cols);
        }
        for (size_t i = 0; i < rows; i++) {
            file.read(reinterpret_cast<char*>(vec[i].data()), cols * sizeof(float));
        }
        file.close();
    } else {
        fprintf(stderr, "failed to open the file for reading");
    }
}