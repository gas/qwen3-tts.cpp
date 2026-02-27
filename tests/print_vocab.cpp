#include <iostream>
#include <fstream>
#include "src/text_tokenizer.h"
#include <ggml.h>
#include <string>
#include <gguf.h>

int main() {
    struct gguf_init_params params = {
        /* .no_alloc   = */ false,
        /* .ctx        = */ NULL,
    };
    struct gguf_context * ctx = gguf_init_from_file("../models/qwen3-tts-0.6b-f16.gguf", params);
    if (!ctx) return 1;
    qwen3_tts::TextTokenizer tokenizer;
    tokenizer.load_from_gguf(ctx);
    
    std::cout << "Token 198: " << tokenizer.decode_token(198) << std::endl;
    std::cout << "Config BOS: " << tokenizer.get_config().bos_token_id << std::endl;
    std::cout << "Config EOS: " << tokenizer.get_config().eos_token_id << std::endl;
    
    auto tokens = tokenizer.encode_for_tts("test");
    for (int t : tokens) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
    return 0;
}
