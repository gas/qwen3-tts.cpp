#include "qwen3_tts.h"
#include "../deps/httplib.h"
#include "../deps/json.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <cstring>
#include <stdexcept>

using json = nlohmann::json;

// Base64 encoding
static const std::string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while((i++ < 3))
            ret += '=';
    }
    return ret;
}

std::vector<uint8_t> create_wav_buffer(const std::vector<float>& audio_data, int sample_rate) {
    int num_samples = audio_data.size();
    int data_size = num_samples * sizeof(int16_t);
    int file_size = 36 + data_size;
    
    std::vector<uint8_t> wav(44 + data_size);
    uint8_t* p = wav.data();
    
    // RIFF chunk
    memcpy(p, "RIFF", 4); p += 4;
    *(int32_t*)p = file_size; p += 4;
    memcpy(p, "WAVE", 4); p += 4;
    
    // fmt chunk
    memcpy(p, "fmt ", 4); p += 4;
    *(int32_t*)p = 16; p += 4;
    *(int16_t*)p = 1; p += 2; // PCM
    *(int16_t*)p = 1; p += 2; // Channels
    *(int32_t*)p = sample_rate; p += 4;
    *(int32_t*)p = sample_rate * sizeof(int16_t); p += 4;
    *(int16_t*)p = sizeof(int16_t); p += 2;
    *(int16_t*)p = 16; p += 2; // bits per sample
    
    // data chunk
    memcpy(p, "data", 4); p += 4;
    *(int32_t*)p = data_size; p += 4;
    
    int16_t* samples = (int16_t*)p;
    for (int i = 0; i < num_samples; i++) {
        float sample = audio_data[i];
        if (sample < -1.0f) sample = -1.0f;
        if (sample > 1.0f) sample = 1.0f;
        samples[i] = (int16_t)(sample * 32767.0f);
    }
    
    return wav;
}

// Global server context
struct ServerContext {
    qwen3_tts::Qwen3TTS engine;
    std::mutex engine_mutex;
    bool loaded = false;
};

int main(int argc, char** argv) {
    std::string model_dir = "./models";
    std::string tts_model_name = "";
    std::string default_voice_ref = "";
    std::string host = "0.0.0.0";
    int port = 8080;
    int chunk_size = 16;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (++i < argc) model_dir = argv[i];
        } else if (arg == "-tts" || arg == "--tts-model") {
            if (++i < argc) tts_model_name = argv[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i < argc) default_voice_ref = argv[i];
        } else if (arg == "-h" || arg == "--host") {
            if (++i < argc) host = argv[i];
        } else if (arg == "-p" || arg == "--port") {
            if (++i < argc) port = std::stoi(argv[i]);
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i < argc) chunk_size = std::stoi(argv[i]);
        }
    }

    ServerContext ctx;
    std::cout << "Loading Qwen3-TTS models from: " << model_dir << " ..." << std::endl;
    if (!ctx.engine.load_models(model_dir, tts_model_name)) {
        std::cerr << "Failed to load models: " << ctx.engine.get_error() << std::endl;
        return 1;
    }
    ctx.loaded = true;
    std::cout << "Models loaded successfully!" << std::endl;

    httplib::Server svr;

    svr.Get("/health", [&ctx](const httplib::Request&, httplib::Response& res) {
        json j = { {"status", "ok"}, {"model_loaded", ctx.loaded} };
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/v1/audio/generations", [&](const httplib::Request& req, httplib::Response& res) {
        json j_res;
        try {
            auto body = json::parse(req.body);
            std::vector<std::string> inputs;

            // Handle both string and array of strings
            if (body.contains("input")) {
                if (body["input"].is_string()) {
                    inputs.push_back(body["input"].get<std::string>());
                } else if (body["input"].is_array()) {
                    for (auto& item : body["input"]) {
                        inputs.push_back(item.get<std::string>());
                    }
                }
            }

            if (inputs.empty()) {
                res.status = 400;
                res.set_content("{\"error\": \"Missing or invalid 'input' field\"}", "application/json");
                return;
            }

            std::string ref_audio = body.value("voice_ref", default_voice_ref); // Existing voice_ref maps to ref_audio
            std::string ref_text = "";
            bool x_vector_only = false;
            
            if (body.contains("voice_ref_audio") && body["voice_ref_audio"].is_string()) {
                ref_audio = body["voice_ref_audio"].get<std::string>();
            }
            
            if (body.contains("voice_ref_text") && body["voice_ref_text"].is_string()) {
                ref_text = body["voice_ref_text"].get<std::string>();
            }
            
            if (body.contains("x_vector_only") && body["x_vector_only"].is_boolean()) {
                x_vector_only = body["x_vector_only"].get<bool>();
            }

            if (ref_text.empty() && !ref_audio.empty()) {
                std::string txt_path = ref_audio;
                size_t dot_pos = txt_path.find_last_of('.');
                if (dot_pos != std::string::npos) {
                    txt_path = txt_path.substr(0, dot_pos) + ".txt";
                    FILE * fp = fopen(txt_path.c_str(), "r");
                    if (fp) {
                        std::string content;
                        char buf[4096];
                        while (size_t bytes = fread(buf, 1, sizeof(buf), fp)) {
                            content.append(buf, bytes);
                        }
                        fclose(fp);
                        while (!content.empty() && (content.back() == '\n' || content.back() == '\r')) {
                            content.pop_back();
                        }
                        ref_text = content;
                        std::cout << "Auto-detected reference text from: " << txt_path << std::endl;
                    }
                }
            }
            int lang_id = 2050; // default to EN

            if (body.contains("language")) {
                std::string lang = body["language"].get<std::string>();
                if (lang == "es") lang_id = 2054;
                else if (lang == "zh") lang_id = 2055;
                // Add more if needed depending on usecase
            }

            qwen3_tts::tts_params params;
            params.language_id = lang_id;
            
            // Check headers, if Accept header asks for raw audio and there is only 1 input
            bool accept_raw = false;
            if (req.has_header("Accept") && req.get_header_value("Accept").find("audio/wav") != std::string::npos && inputs.size() == 1) {
                accept_raw = true;
            }

            std::vector<qwen3_tts::tts_result> results;
            {
                // Synchronize access to deep TTS engine state
                std::lock_guard<std::mutex> lock(ctx.engine_mutex);
                
                for (size_t i = 0; i < inputs.size(); i += chunk_size) {
                    size_t end_idx = std::min(i + chunk_size, inputs.size());
                    std::vector<std::string> chunk_inputs(inputs.begin() + i, inputs.begin() + end_idx);
                    
                    size_t original_chunk_size = chunk_inputs.size();
                    while (chunk_inputs.size() < (size_t)chunk_size) {
                        chunk_inputs.push_back(chunk_inputs.back());
                    }
                    
                    std::vector<qwen3_tts::tts_result> chunk_results;
            
                    if (ref_audio.empty()) {
                        chunk_results = ctx.engine.synthesize_batch(chunk_inputs, "", "", false, params);
                    } else {
                        chunk_results = ctx.engine.synthesize_batch(chunk_inputs, ref_audio, ref_text, x_vector_only, params);
                    }
                    results.insert(results.end(), chunk_results.begin(), chunk_results.begin() + original_chunk_size);
                }
            }

            if (accept_raw) {
                if (!results[0].success) {
                    res.status = 500;
                    res.set_content("{\"error\": \"Engine failed\"}", "application/json");
                } else {
                    auto wav_bytes = create_wav_buffer(results[0].audio, results[0].sample_rate);
                    res.set_content(reinterpret_cast<const char*>(wav_bytes.data()), wav_bytes.size(), "audio/wav");
                }
                return;
            }

            j_res["generations"] = json::array();
            for (size_t i = 0; i < results.size(); ++i) {
                json item;
                item["success"] = results[i].success;
                if (results[i].success) {
                    auto wav_bytes = create_wav_buffer(results[i].audio, results[i].sample_rate);
                    item["audio_base64"] = base64_encode(wav_bytes.data(), wav_bytes.size());
                    item["sample_rate"] = results[i].sample_rate;
                } else {
                    item["error"] = results[i].error_msg;
                }
                j_res["generations"].push_back(item);
            }
            res.set_content(j_res.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(std::string("{\"error\": \"") + e.what() + "\"}", "application/json");
        }
    });

    std::cout << "Starting server on " << host << ":" << port << "..." << std::endl;
    svr.listen(host, port);

    return 0;
}
