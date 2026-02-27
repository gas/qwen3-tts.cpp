#include "voice_profile.h"

#include <fstream>
#include <iostream>

namespace qwen3_tts {

bool load_voice_profile(const std::string & path, voice_profile & profile) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open voice profile file: " << path << std::endl;
        return false;
    }

    // Read magic header
    char magic[4];
    if (!file.read(magic, 4) || std::string(magic, 4) != "Q3VP") {
        std::cerr << "Invalid voice profile: Missing 'Q3VP' header in " << path << std::endl;
        return false;
    }

    // Read metadata
    if (!file.read(reinterpret_cast<char*>(&profile.version), sizeof(uint32_t)) ||
        !file.read(reinterpret_cast<char*>(&profile.sample_rate), sizeof(uint32_t)) ||
        !file.read(reinterpret_cast<char*>(&profile.spk_dim), sizeof(uint32_t)) ||
        !file.read(reinterpret_cast<char*>(&profile.num_frames), sizeof(uint32_t)) ||
        !file.read(reinterpret_cast<char*>(&profile.num_quantizers), sizeof(uint32_t))) {
        std::cerr << "Invalid voice profile: Failed to read metadata in " << path << std::endl;
        return false;
    }

    if (profile.version != 1) {
        std::cerr << "Unsupported voice profile version: " << profile.version << std::endl;
        return false;
    }

    // Read speaker embedding
    profile.speaker_embedding.resize(profile.spk_dim);
    if (!file.read(reinterpret_cast<char*>(profile.speaker_embedding.data()), profile.spk_dim * sizeof(float))) {
        std::cerr << "Invalid voice profile: Failed to read speaker embedding in " << path << std::endl;
        return false;
    }

    // Read audio codes
    size_t total_codes = static_cast<size_t>(profile.num_frames) * profile.num_quantizers;
    profile.audio_codes.resize(total_codes);
    if (!file.read(reinterpret_cast<char*>(profile.audio_codes.data()), total_codes * sizeof(int32_t))) {
        std::cerr << "Invalid voice profile: Failed to read audio codes in " << path << std::endl;
        return false;
    }

    return true;
}

} // namespace qwen3_tts
