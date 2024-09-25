#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <dirent.h>
#include <map>
#include "clip.hpp"
#include "tokenizer.hpp"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

enum similarity_measure {
    COSINE_SIMILARITY,
    EUCLIDEAN_DISTANCE,
};

template <class VisionModel, class TextModel>
class Pipeline {
public:
    Pipeline(ggml_type type, CLIPVersion version, ClipVisionParam vision_params, ClipTextModelParam text_params)
        : type(type), version(version), vision_params(vision_params), text_params(text_params){};
    std::vector<std::pair<std::string, float>> zeros_shot_image_classify(std::string img_path, std::vector<std::string> labels, int topk = 3);
    std::vector<std::pair<std::string, float>> text_search_image(std::string& text, std::vector<std::string>& image_path, bool read_local_embeding, bool save_embeding, std::string embeding_path = "./emdedings.bin", int topk = 3);

    void model_load(std::string path);
    std::vector<std::string> find_image_path(const std::string& directory);

private:
    float compute_similarity(float* v0, float* v1, int n, similarity_measure mearture = COSINE_SIMILARITY);
    bool is_imagefile(const std::string& filename);

    ggml_type type;
    CLIPVersion version;
    ClipVisionParam vision_params;
    ClipTextModelParam text_params;
    std::unique_ptr<ClipModel<VisionModel, TextModel>> clip_model;
};

template <class VisionModel, class TextModel>
void Pipeline<VisionModel, TextModel>::model_load(std::string model_path) {
    const char* sys_info = sd_get_system_info();
    printf("%s\n", sys_info);
    ggml_backend_t backend = NULL;
#ifdef GGML_USE_METAL
    backend = ggml_backend_metal_init();
#endif
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }

    clip_model = std::make_unique<ClipModel<VisionModel, TextModel>>(backend, type, text_params, vision_params, version);
    clip_model->load_weight(model_path);
}

template <class VisionModel, class TextModel>
std::vector<std::pair<std::string, float>> Pipeline<VisionModel, TextModel>::zeros_shot_image_classify(std::string img_path, std::vector<std::string> labels, int topk) {
    int width, height, c;
    unsigned char* data = stbi_load(img_path.c_str(), &width, &height, &c, 3);
    image_u8_t image_u8{(uint32_t)width, (uint32_t)height, (uint32_t)c, data};
    printf("img_path:%s,width=%d,height=%d,c=%d\n", img_path.c_str(), width, height, c);

    std::vector<float> vis_vec_out;
    clip_model->get_image_features(image_u8, vis_vec_out);

    std::vector<std::vector<float>> text_vec;
    for (int i = 0; i < labels.size(); i++) {
        std::vector<float> vec;
        std::string s = labels[i];
        clip_model->get_text_features(s, vec);
        text_vec.push_back(vec);
    }

    std::vector<std::pair<std::string, float>> scores;
    for (int i = 0; i < labels.size(); i++) {
        float s0 = compute_similarity(vis_vec_out.data(), text_vec[i].data(), text_vec[i].size(), COSINE_SIMILARITY);
        scores.push_back({labels[i], s0});
    }

    sort(scores.begin(), scores.end(), [&](std::pair<std::string, float>& p0, std::pair<std::string, float>& p1) -> bool {
        return p1.second < p0.second;
    });

#ifdef CLIP_DEBUG
    std::cout << std::left << std::setw(20) << "label" << "   scores  " << std::endl;
    int k = 0;
    for (auto& v : scores) {
        std::cout << std::left << std::setw(20) << v.first << "   :" << v.second << std::endl;
        k += 1;
        if (k >= topk)
            break;
    }
#endif

    free(image_u8.data);
    std::vector<std::pair<std::string, float>> score_topk(scores.begin(), scores.begin() + topk);
    return score_topk;
}

template <class VisionModel, class TextModel>
float Pipeline<VisionModel, TextModel>::compute_similarity(float* v0, float* v1, int n, similarity_measure mearture) {
    float score = 0.f;
    if (mearture == COSINE_SIMILARITY) {
        for (int i = 0; i < n; i++) {
            score += v0[i] * v1[i];
        }
    } else if (mearture == EUCLIDEAN_DISTANCE) {
        for (int i = 0; i < n; i++) {
            score += (v0[i] - v1[i]) * (v0[i] - v1[i]);
        }
        score = 1.0 / (sqrtf(score) + 1e-5);
    }
    return score;
}

template <class VisionModel, class TextModel>
bool Pipeline<VisionModel, TextModel>::is_imagefile(const std::string& filename) {
    static const std::vector<std::string> imageExtensions = {".png", ".jpg", ".jpeg"};
    for (const auto& extension : imageExtensions) {
        if (filename.size() >= extension.size() &&
            filename.compare(filename.size() - extension.size(), extension.size(), extension) == 0) {
            return true;
        }
    }
    return false;
}

template <class VisionModel, class TextModel>
std::vector<std::string> Pipeline<VisionModel, TextModel>::find_image_path(const std::string& directory) {
    DIR* dir;
    struct dirent* entry;
    std::vector<std::string> img_path;
    if ((dir = opendir(directory.c_str())) != nullptr) {
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename(entry->d_name);
            if (is_imagefile(filename)) {
                img_path.push_back(directory + "/" + filename);
            }
        }
        closedir(dir);
    } else {
        fprintf(stderr, "Failed to open directory\n");
    }
    return img_path;
}

template <class VisionModel, class TextModel>
std::vector<std::pair<std::string, float>> Pipeline<VisionModel, TextModel>::text_search_image(std::string& text,
                                                                                               std::vector<std::string>& image_path,
                                                                                               bool read_local_embeding,
                                                                                               bool save_embeding,
                                                                                               std::string embeding_path,
                                                                                               int topk) {
    std::vector<std::vector<float>> vis_out;
    if (!read_local_embeding) {
        unsigned char* data;
        for (int i = 0; i < image_path.size(); i++) {
            std::vector<float> vis_vec_out;
            int width, height, c;
            data = stbi_load(image_path[i].c_str(), &width, &height, &c, 3);
            image_u8_t image_u8{(uint32_t)width, (uint32_t)height, (uint32_t)c, data};
            printf("img_path:%s,width=%d,height=%d,c=%d\n", image_path[i].c_str(), width, height, c);

            clip_model->get_image_features(image_u8, vis_vec_out);
            vis_out.push_back(vis_vec_out);
        }
        free(data);
        if (save_embeding) {
            VectorIO vec_io;
            vec_io.vec_store(embeding_path, vis_out);
        }
    } else {
        assert(file_exists(embeding_path));
        VectorIO vec_io;
        vec_io.vec_load(embeding_path, vis_out);
    }
    std::vector<float> text_vec_out;
    clip_model->get_text_features(text, text_vec_out);

    std::vector<std::pair<std::string, float>> scores;
    for (int i = 0; i < vis_out.size(); i++) {
        float s0 = compute_similarity(vis_out[i].data(), text_vec_out.data(), text_vec_out.size(), COSINE_SIMILARITY);
        scores.push_back({image_path[i], s0});
    }

    sort(scores.begin(), scores.end(), [&](std::pair<std::string, float>& p0, std::pair<std::string, float>& p1) -> bool {
        return p1.second < p0.second;
    });

#ifdef CLIP_DEBUG
    std::cout << "label:" << text << std::right << std::setw(72) << "   scores  " << std::endl;
    int k = 0;
    for (auto& v : scores) {
        std::cout << std::left << std::setw(80) << v.first << "   :" << v.second << std::endl;
        k += 1;
        if (k >= topk)
            break;
    }
#endif
    std::vector<std::pair<std::string, float>> score_topk(scores.begin(), scores.begin() + topk);
    return score_topk;
}
