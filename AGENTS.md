# AGENTS.md

Coding conventions and architecture guide for AI agents working on this codebase.

## Project Overview

`qwen3-tts.cpp` is a pure C++17 implementation of the Qwen3-TTS text-to-speech pipeline using GGML. It converts text to speech through four stages: tokenization, speaker encoding, transformer code generation, and vocoder decoding. The engine supports **Zero-shot inference**, **Voice Cloning (ICL & X-Vector Only)**, and handles **safe batch chunking and pagination** for massive text generations protecting GPU VRAM.

## Repository Structure

```
qwen3-tts.cpp/
  src/                          # C++ source files
    main.cpp                    # CLI entry point
    server.cpp                  # HTTP REST Server (httplib + json)
    qwen3_tts_c_api.cpp         # Dynamic C API bindings (libqwen3-tts.so)
    qwen3_tts.{h,cpp}          # Full pipeline orchestration
    tts_transformer.{h,cpp}    # TTS transformer (talker + code predictor)
    text_tokenizer.{h,cpp}     # BPE text tokenizer
    audio_tokenizer_encoder.{h,cpp}  # ECAPA-TDNN speaker embedding extraction
    audio_tokenizer_decoder.{h,cpp}  # WavTokenizer vocoder
    gguf_loader.{h,cpp}        # GGUF model loading
    voice_profile.cpp          # Voice profile .q3vp binary parser
  tests/                        # Component tests
    test_codebook.cpp
    test_vq_only.cpp
    test_tokenizer.cpp
    test_encoder.cpp
    test_transformer.cpp        # Deterministic reference comparison
    test_decoder.cpp
    print_vocab.cpp
  scripts/                      # Python utilities
    convert_tts_to_gguf.py      # HuggingFace -> GGUF converter (TTS model)
    convert_tokenizer_to_gguf.py # HuggingFace -> GGUF converter (vocoder)
    extract_voice_profile.py    # Python script to generate .q3vp voice clones
    benchmark_modalities.py     # Evaluates RTF and memory on the inferencing modalities
    generate_deterministic_reference.py  
    run_all_tests.sh            # Test runner
  docs/                         # Project technical documentation
    q3vp_format.md              # Voice Profile format specification
  reference/                    # Reference data (*.bin gitignored, *.json tracked)
  models/                       # GGUF models (gitignored)
  CMakeLists.txt
```

## Build System

- **CMake 3.14+** with C++17
- GGML is vendored under `./ggml` and linked from `./ggml/build/src`
- Build GGML first: `cmake -S ggml -B ggml/build -DGGML_METAL=ON && cmake --build ggml/build -j4`
- Build project: `cmake -S . -B build && cmake --build build -j4`
- Timing build: `cmake -S . -B build -DQWEN3_TTS_TIMING=ON && cmake --build build -j4`
- GGML headers are in `./ggml/include`

It produces 3 main targets:
1. `qwen3-tts-cli` (Console entry)
2. `libqwen3-tts.so` / `dll` (Shared C API library)
3. `qwen3-tts-server` (HTTP REST endpoint)

## Important Milestones & Architectures

### 1. Batching and Pagination (OOM Protection)
The engine has robust logic to prevent VRAM Out-of-Memory (OOM) errors during bulk processing.
Using `-b <N>` (Batch Size) in the CLI or through the API restricts the simultaneous KV-Cache blocks requested to the GPU. 
During inference, the array of strings is paginated into chunks. If a chunk is smaller than the requested cache dimension, it is safely padded using **Dummy Batch Padding** (repeating the last sequence) to avoid runtime segmentation faults in `ggml_cpy`.
The REST Server utilizes a global `std::mutex` to queue concurrent requests avoiding race conditions locking VRAM.

### 2. Voice Profiling (Q3VP & Voice Cloning)
Voice cloning allows mimicking a speaker's timbre and prosody.
`extract_voice_profile.py` uses PyTorch to process a reference Audio and output a `.q3vp` (Qwen3 Voice Profile) binary containing two layers:
- Payload 1: **X-Vector** / Speaker Embedding (1024-dim, static timbre form)
- Payload 2: **Acoustic Codes** / In-Context Learning (Prosody condition tokens)

There are two primary ways of generating synthetic clones in `qwen3-tts.cpp`:
- **Full ICL (In-Context Learning)**: `-r <file.q3vp> -p "Exact transcript of audio"`. Extremely faithful but prone to starting high-pitch instability inside the Qwen3 transformer if the `ref_text` misaligns with the reference tokens.
- **X-Vector Only Mode**: `-r <file.q3vp> -x` (or `"x_vector_only": true` via JSON). Completely omits the Acoustic Codes, parsing strictly the timbre. This avoids the transcript dependency and generates 100% pitch-stable outputs natively skipping the ICL block. TTFT (Time To First Token) latency is also vastly improved in this mode.

### 3. Dynamic C API
The core logic resides functionally isolated in `Qwen3TTS::synthesize_batch` which is interfaced through standard C bindings in `qwen3_tts_c_api.cpp`, enabling direct loading via FFI bridges from other languages like Python (ctypes), Go, or Node without launching shell child processes.

## Coding Conventions

### C++ Style

- C++17 standard, no exceptions, no RTTI inside core. 
- API Wrappers (`main.cpp`, `qwen3_tts_c_api.cpp`) capture std::exceptions strictly translating them to string `error_msg_`s.
- Memory: GGML contexts own tensor memory; use `ggml_free()` for cleanup.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for macros.
- All public types in `qwen3_tts` namespace.

### GGML Patterns

Every forward pass follows this pattern:

```cpp
// 1. Build computation graph
struct ggml_cgraph * gf = build_xxx_graph(...);

// 2. Allocate graph memory
ggml_backend_sched_alloc_graph(state_.sched, gf);

// 3. Set input tensors
struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "input_name");
ggml_backend_tensor_set(inp, data, 0, size);

// 4. Compute
ggml_backend_sched_graph_compute(state_.sched, gf);

// 5. Get output tensors
struct ggml_tensor * out = ggml_graph_get_tensor(gf, "output_name");
ggml_backend_tensor_get(out, output_data, 0, size);

// 6. Reset scheduler
ggml_backend_sched_reset(state_.sched);
```

Important: `ggml_cast` to F32 is needed before `ggml_mul_mat` when weight tensors are F16 (specifically `ffn_down` in both talker and code predictor layers).

## Performance Profile (Based on latest macOS/ROCm metrics)

The **Code Predictor transformer** remains the primary bottleneck (~70% of generation time) because it runs 15 sequential forward passes per audio frame (1 prefill + 14 cascaded autoregressive steps for residual quantizers). 
Graph building/allocations and data I/O represent `<1%` overhead overall.

**Baseline Metrics (Qwen3-TTS 1.7B-Base):**
- **Zero-Shot generation**: Fastest baseline. Generation speeds scale linearly against the word count.
- **X-Vector Only (`-x`)**: Near-identical runtime and latency compared to Zero-Shot (`~6.7 RTF` on fallback GPUs). Imposes a strict `~1.2 GB VRAM` peak memory footprint on `1.7B-f16` without penalizing context width caching.
- **Full ICL Clone (`-p <text> -r <.q3vp>`)**: Considerably slower Time-To-First-Token (TTFT) and slower sequence iteration due to heavy relative positional encoding (RoPE) prepended by the loaded acoustic codes inside the initial context window. Expect up to +40% longer initialization times over X-Vector configurations.
