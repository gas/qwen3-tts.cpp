#include "qwen3_tts_c_api.h"
#include "qwen3_tts.h"

#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

struct qwen3_tts_context {
    qwen3_tts::Qwen3TTS engine;
    std::string last_error;
};

// Allocate a C string dynamically
static char* c_strdup(const std::string& str) {
    if (str.empty()) return nullptr;
    char* copy = (char*)malloc(str.length() + 1);
    std::memcpy(copy, str.c_str(), str.length() + 1);
    return copy;
}

qwen3_tts_c_params qwen3_tts_default_params(void) {
    qwen3_tts::tts_params cpp_params;
    qwen3_tts_c_params p;
    p.temperature = cpp_params.temperature;
    p.top_k = cpp_params.top_k;
    p.top_p = cpp_params.top_p;
    p.max_audio_tokens = cpp_params.max_audio_tokens;
    p.repetition_penalty = cpp_params.repetition_penalty;
    p.language_id = cpp_params.language_id;
    p.n_threads = cpp_params.n_threads;
    return p;
}

qwen3_tts_context* qwen3_tts_init(void) {
    return new qwen3_tts_context();
}

bool qwen3_tts_load_models(qwen3_tts_context* ctx, const char* model_dir, const char* tts_model_name) {
    if (!ctx || !model_dir) return false;
    std::string dir_str(model_dir);
    std::string model_str = tts_model_name ? std::string(tts_model_name) : "";
    
    bool ok = ctx->engine.load_models(dir_str, model_str);
    if (!ok) {
        ctx->last_error = ctx->engine.get_error();
    }
    return ok;
}

qwen3_tts_batch_result* qwen3_tts_synthesize_batch(
    qwen3_tts_context* ctx, 
    const char** texts, 
    size_t num_texts, 
    const char* reference_audio_path, 
    const char* reference_text,
    bool x_vector_only,
    qwen3_tts_c_params params) 
{
    if (!ctx || !texts || num_texts == 0) return nullptr;

    try {
        std::vector<std::string> text_vec;
        text_vec.reserve(num_texts);
        for (size_t i = 0; i < num_texts; ++i) {
            if (texts[i]) {
                text_vec.push_back(std::string(texts[i]));
            } else {
                return nullptr; // Invalid input array
            }
        }

        std::string ref_audio = reference_audio_path ? std::string(reference_audio_path) : "";
        std::string ref_text = reference_text ? std::string(reference_text) : "";

        qwen3_tts::tts_params cpp_params;
        cpp_params.temperature = params.temperature;
        cpp_params.top_k = params.top_k;
        cpp_params.top_p = params.top_p;
        cpp_params.max_audio_tokens = params.max_audio_tokens;
        cpp_params.repetition_penalty = params.repetition_penalty;
        cpp_params.language_id = params.language_id;
        cpp_params.n_threads = params.n_threads;

        std::vector<qwen3_tts::tts_result> cpp_results = ctx->engine.synthesize_batch(text_vec, ref_audio, ref_text, x_vector_only, cpp_params);

        qwen3_tts_batch_result* out = (qwen3_tts_batch_result*)malloc(sizeof(qwen3_tts_batch_result));
        out->num_results = cpp_results.size();
        out->results = (qwen3_tts_result*)calloc(out->num_results, sizeof(qwen3_tts_result));

        for (size_t i = 0; i < out->num_results; ++i) {
            out->results[i].success = cpp_results[i].success;
            out->results[i].sample_rate = cpp_results[i].sample_rate;
            
            if (cpp_results[i].success) {
                out->results[i].audio_len = cpp_results[i].audio.size();
                out->results[i].audio_data = (float*)malloc(cpp_results[i].audio.size() * sizeof(float));
                std::memcpy(out->results[i].audio_data, cpp_results[i].audio.data(), cpp_results[i].audio.size() * sizeof(float));
                out->results[i].error_msg = nullptr;
            } else {
                out->results[i].audio_len = 0;
                out->results[i].audio_data = nullptr;
                out->results[i].error_msg = c_strdup(cpp_results[i].error_msg);
            }
        }

        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    }
}

const char* qwen3_tts_get_last_error(qwen3_tts_context* ctx) {
    if (!ctx) return nullptr;
    return ctx->last_error.c_str();
}

void qwen3_tts_free_batch_result(qwen3_tts_batch_result* result) {
    if (!result) return;
    if (result->results) {
        for (size_t i = 0; i < result->num_results; ++i) {
            if (result->results[i].audio_data) {
                free(result->results[i].audio_data);
            }
            if (result->results[i].error_msg) {
                free((void*)result->results[i].error_msg);
            }
        }
        free(result->results);
    }
    free(result);
}

void qwen3_tts_free(qwen3_tts_context* ctx) {
    if (ctx) {
        delete ctx;
    }
}
