#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "pipeline.hpp"
using namespace std;

const char* clip_mode_name[] = {
    "zeros_shot_image_classification",
    "text_search_image",
};

typedef enum clip_mode {
    ZEROS_SHOT_IMAGE_CLASSIFICATION = 0,
    TEXT_SEARCH_IMAGE,
    CLIP_MODE_COUNT
} clip_mode;

typedef struct text_search_img {
    bool read_local_embeding = false;
    bool save_embeding       = true;
    string save_path         = "./embedings.bin";
    string text;
    string img_directory;
} text_search_img_params;

typedef struct zeros_shot_classification {
    string img_path;
    string label_path;

} zeros_shot_classification_params;

typedef struct clip_params {
    ggml_type model_type = GGML_TYPE_Q8_0;
    clip_mode mode       = TEXT_SEARCH_IMAGE;
    CLIPVersion version;
    string model_path;
    zeros_shot_classification_params classify_params;
    text_search_img_params text_img_params;
} clip_params;

void print_help(int argc, char** argv) {
    printf("Usage: %s [options]\n", argv[0]);
    printf("\nOptions:\n");
    printf("  -h, --help: Show this message and exit\n");
    printf("  -m <path>, --model <path>: path to model\n");
    printf(
        "  --model_version: the clip model version [openai_clip_vit_base_patch32,openai_clip_vit_large_pathc14, \
            openai_clip_vit_large_patch14_336,openai_clip_vit_base_patch16]\n");
    printf("  --mode: clip mode must be one of [zeros_shot_image_classification,text_search_image],default = text_search_image\n");
    printf("  --model_type: clip model type choose from [f32,f16,q8_0],default = GGML_TYPE_Q8_0 \n");
    printf("  --image_path <path>: path to an image file for zeros_shot_image_classify \n");
    printf("  --label_path:txt file for zeros_shot_image_classification\n");
    printf("  --text:  the text for text_search_image\n");
    printf("  --img_directory: the image directory for text_search_image\n");
}

bool params_parse(int argc, char** argv, clip_params& params) {
    bool invalid_arg = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = std::string(argv[i]);
        if (arg == "-m" || arg == "--model") {
            params.model_path = argv[++i];
        } else if (arg == "--mode") {
            const char* model_mode = argv[++i];
            int mode_found         = -1;
            for (int d = 0; d < CLIP_MODE_COUNT; d++) {
                if (!strcmp(model_mode, clip_mode_name[d])) {
                    mode_found = d;
                }
            }
            if (mode_found == -1) {
                fprintf(stderr,
                        "error: invalid mode %s, must be one of [zeros_shot_image_classify,text_search_image]\n",
                        model_mode);
                exit(1);
            }
            params.mode = (clip_mode)mode_found;

        } else if (arg == "--image_path") {
            params.classify_params.img_path = argv[++i];

        } else if (arg == "--model_type") {
            string model_type = argv[++i];
            if (model_type == "f32") {
                params.model_type = GGML_TYPE_F32;
            } else if (model_type == "f16") {
                params.model_type = GGML_TYPE_F16;
            } else if (model_type == "q8_0") {
                params.model_type = GGML_TYPE_Q8_0;
            } else {
                fprintf(stderr, "error model type:%s\n", model_type.c_str());
                exit(0);
            }
        } else if (arg == "--model_version") {
            const char* model_version = argv[++i];
            int version_found         = -1;
            for (int d = 0; d < VERSION_COUNT; d++) {
                if (!strcmp(model_version, clip_version[d])) {
                    version_found = d;
                }
            }
            if (version_found == -1) {
                fprintf(stderr,
                        "error: invalid mode %s, must be one of [zeros_shot_image_classify,text_search_image]\n",
                        model_version);
                exit(1);
            }
            params.version = (CLIPVersion)version_found;
        } else if (arg == "--label_path") {
            params.classify_params.label_path = argv[++i];
        } else if (arg == "--text") {
            params.text_img_params.text = argv[++i];
        } else if (arg == "--img_directory") {
            params.text_img_params.img_directory = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_help(argc, argv);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_help(argc, argv);
            exit(1);
        }
    }
    return true;
}

void print_result(string label, vector<pair<string, float>> scores) {
    std::cout << "label:"<<label << std::endl;
    for (auto& t : scores) {
        std::cout << std::left << std::setw(70) << t.first << "   :" << t.second << std::endl;
    }
}

template <typename VisionModel, typename TextModel>
void run_example(clip_params params) {
    ggml_type type      = params.model_type;
    CLIPVersion version = params.version;
    clip_mode mode      = params.mode;
    string model_path   = params.model_path;

    ClipVisionParam vision_params(version);
    ClipTextModelParam text_params(version);

    std::unique_ptr<Pipeline<VisionModel, TextModel>> pipeline(new Pipeline<VisionModel, TextModel>(type, version, vision_params, text_params));
    pipeline->model_load(model_path);

    vector<std::pair<string, float>> scores;
    if (params.mode == ZEROS_SHOT_IMAGE_CLASSIFICATION) {
        string label_path = params.classify_params.label_path;
        string img_path   = params.classify_params.img_path;
        vector<string> label_vec;
        std::ifstream file(label_path);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                label_vec.push_back(line);
            }
            file.close();
        } else {
            fprintf(stderr, "fail to open file\n");
        }
        scores = pipeline->zeros_shot_image_classify(img_path, label_vec);
        print_result(img_path, scores);
    } else if (params.mode == TEXT_SEARCH_IMAGE) {
        bool read_local_embeding = params.text_img_params.read_local_embeding;
        bool save_embeding       = params.text_img_params.save_embeding;
        string save_path         = params.text_img_params.save_path;
        string text              = params.text_img_params.text;
        string img_directory     = params.text_img_params.img_directory;

        std::stringstream ss(text);
        std::vector<std::string> text_vec;
        std::string token;
        while (std::getline(ss, token, ',')) {
            text_vec.push_back(token);
        }
        vector<string> image_path = pipeline->find_image_path(img_directory);
        for (int i = 0; i < text_vec.size(); i++) {
            scores              = pipeline->text_search_image(text_vec[i], image_path, read_local_embeding, save_embeding, save_path, 3);
            read_local_embeding = true;
            print_result(text_vec[i], scores);
        }
    }
}

int main(int argc, char** argv) {
    clip_params params;
    params_parse(argc, argv, params);

    CLIPVersion version = params.version;
    if (version == OFASYS_CHINESE_CLIP_VIT_HUGE_PATCH14 || version == OFASYS_CHINESE_CLIP_VIT_LARGE_PATCH14_336 ||
        version == OFASYS_CHINESE_CLIP_VIT_BASE_PATCH16 || version == OFASYS_CHINESE_CLIP_VIT_LARGE_PATCH14) {
        run_example<ClipVisionModel, ClipChineseTextModel>(params);

    } else {
        run_example<ClipVisionModel, ClipTextModel>(params);
    }
    return 0;
}