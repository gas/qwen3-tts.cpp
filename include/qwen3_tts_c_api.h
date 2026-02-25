#ifndef QWEN3_TTS_C_API_H
#define QWEN3_TTS_C_API_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#if defined(_WIN32)
#if defined(QWEN3_TTS_SHARED)
#define QWEN3_TTS_API __declspec(dllexport)
#else
#define QWEN3_TTS_API __declspec(dllimport)
#endif
#else
#define QWEN3_TTS_API __attribute__ ((visibility ("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to the TTS context
typedef struct qwen3_tts_context qwen3_tts_context;

// Struct to hold individual generation results
typedef struct {
    float* audio_data;       // Raw PCM float32 audio data
    size_t audio_len;        // Number of samples
    int sample_rate;         // Typically 24000
    bool success;            // true if generated successfully
    const char* error_msg;   // Error message if success is false, null otherwise
} qwen3_tts_result;

// Struct to hold batch results
typedef struct {
    qwen3_tts_result* results; // Array of results
    size_t num_results;        // Number of results (equals number of input texts)
} qwen3_tts_batch_result;

// Parameters for synthesis
typedef struct {
    float temperature;
    int top_k;
    float top_p;
    int max_audio_tokens;
    float repetition_penalty;
    int language_id;         // E.g. 2050 for English, 2054 for Spanish, etc.
    int n_threads;
} qwen3_tts_c_params;

/**
 * Get default synthesis parameters
 */
QWEN3_TTS_API qwen3_tts_c_params qwen3_tts_default_params(void);

/**
 * Initialize a new TTS context.
 */
QWEN3_TTS_API qwen3_tts_context* qwen3_tts_init(void);

/**
 * Load models into the context.
 * @param ctx The TTS context
 * @param model_dir The directory containing the models
 * @param tts_model_name The filename of the TTS model (can be NULL or empty for default)
 * @return true on success, false on failure (use qwen3_tts_get_last_error to get details)
 */
QWEN3_TTS_API bool qwen3_tts_load_models(qwen3_tts_context* ctx, const char* model_dir, const char* tts_model_name);

/**
 * Synthesize a batch of texts.
 * @param ctx The TTS context
 * @param texts Array of text strings
 * @param num_texts Number of elements in the texts array
 * @param reference_audio_path Path to a reference WAV file for voice cloning (can be NULL or empty for default speaker)
 * @param params Synthesis parameters
 * @return A qwen3_tts_batch_result struct containing the generated audio. Must be freed with qwen3_tts_free_batch_result.
 */
QWEN3_TTS_API qwen3_tts_batch_result* qwen3_tts_synthesize_batch(
    qwen3_tts_context* ctx, 
    const char** texts, 
    size_t num_texts, 
    const char* reference_audio_path, 
    qwen3_tts_c_params params);

/**
 * Return the last internal error message if an operation failed.
 */
QWEN3_TTS_API const char* qwen3_tts_get_last_error(qwen3_tts_context* ctx);

/**
 * Free the memory allocated for a batch result.
 */
QWEN3_TTS_API void qwen3_tts_free_batch_result(qwen3_tts_batch_result* result);

/**
 * Free the TTS context and destroy all associated models.
 */
QWEN3_TTS_API void qwen3_tts_free(qwen3_tts_context* ctx);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_TTS_C_API_H
