cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX HIP)


# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fgpu-rdc")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Timing instrumentation option
option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

# Find threads
find_package(Threads REQUIRED)
find_package(rocprim REQUIRED CONFIG)
find_package(rocthrust REQUIRED CONFIG)
find_package(hipthreads REQUIRED)

# GGML library paths
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
set(GGML_BUILD_DIR "${GGML_DIR}/build")

# Text tokenizer library
add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
    src/gguf_loader.cpp
)
target_include_directories(text_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(text_tokenizer PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(text_tokenizer PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)

# TTS Transformer library (GGML-based + optional CoreML bridge)
set(TTS_TRANSFORMER_SOURCES
    src/gguf_loader.cpp
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC
    ${TTS_TRANSFORMER_SOURCES}
)
target_include_directories(tts_transformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(tts_transformer PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(tts_transformer PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()

# Audio tokenizer encoder library (GGML-based)
add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(audio_tokenizer_encoder PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(audio_tokenizer_encoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)

# Audio tokenizer decoder library (GGML-based)
add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
    src/gguf_loader.cpp
)
set_source_files_properties(src/audio_tokenizer_decoder.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(audio_tokenizer_decoder PROPERTIES 
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)
target_include_directories(audio_tokenizer_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(audio_tokenizer_decoder PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(audio_tokenizer_decoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
    hipthreads::hipthreads
)

# Qwen3 TTS library (full pipeline)
add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
    src/voice_profile.cpp
)
set_source_files_properties(src/qwen3_tts.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(qwen3_tts PROPERTIES 
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    Threads::Threads
    roc::rocthrust
)

# Qwen3 TTS Dynamic Library (C API)
add_library(qwen3_tts_shared SHARED
    src/qwen3_tts_c_api.cpp
)
set_source_files_properties(src/qwen3_tts_c_api.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(qwen3_tts_shared PROPERTIES 
    OUTPUT_NAME "qwen3-tts"
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
    POSITION_INDEPENDENT_CODE ON
)
target_compile_definitions(qwen3_tts_shared PRIVATE QWEN3_TTS_SHARED)
target_include_directories(qwen3_tts_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_tts_shared PRIVATE
    qwen3_tts
)

# CLI executable
add_executable(qwen3-tts-cli
    src/main.cpp
)
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(qwen3-tts-cli PROPERTIES 
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
)

# HTTP Server executable
add_executable(qwen3-tts-server
    src/server.cpp
)
set_source_files_properties(src/server.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(qwen3-tts-server PROPERTIES 
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)
target_include_directories(qwen3-tts-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/deps
)
target_link_libraries(qwen3-tts-server PRIVATE
    qwen3_tts
)

# Test executable for tokenizer
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_tokenizer_encoder
    Threads::Threads
)

# Test executable for transformer
add_executable(test_transformer
    tests/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE
    tts_transformer
    Threads::Threads
)

# Test executable for audio decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
set_source_files_properties(tests/test_decoder.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(test_decoder PROPERTIES 
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
    Threads::Threads
    roc::rocthrust
)

# Test executable for hipThreads experimental revival
add_executable(test_hipthreads
    tests/test_hipthreads.hip
)
set_source_files_properties(tests/test_hipthreads.hip PROPERTIES LANGUAGE HIP)
set_target_properties(test_hipthreads PROPERTIES 
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)
target_include_directories(test_hipthreads PRIVATE
    ${GGML_DIR}/include
)
target_link_libraries(test_hipthreads PRIVATE
    Threads::Threads
    hipthreads::hipthreads
    roc::rocthrust
)

# Install targets
install(TARGETS text_tokenizer tts_transformer audio_tokenizer_encoder audio_tokenizer_decoder qwen3_tts qwen3-tts-cli
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES 
    src/gguf_loader.h 
    src/text_tokenizer.h 
    src/tts_transformer.h 
    src/coreml_code_predictor.h
    src/audio_tokenizer_encoder.h 
    src/audio_tokenizer_decoder.h 
    src/qwen3_tts.h
    DESTINATION include
)

# CTest support
enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME encoder_test
    COMMAND test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME transformer_test
    COMMAND test_transformer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME decoder_test
    COMMAND test_decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
