import ctypes
import os
import struct
import wave
import sys

# Define C structures mapping
class Qwen3TTSParams(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("max_audio_tokens", ctypes.c_int),
        ("repetition_penalty", ctypes.c_float),
        ("language_id", ctypes.c_int),
        ("n_threads", ctypes.c_int),
    ]

class Qwen3TTSResult(ctypes.Structure):
    _fields_ = [
        ("audio_data", ctypes.POINTER(ctypes.c_float)),
        ("audio_len", ctypes.c_size_t),
        ("sample_rate", ctypes.c_int),
        ("success", ctypes.c_bool),
        ("error_msg", ctypes.c_char_p),
    ]

class Qwen3TTSBatchResult(ctypes.Structure):
    _fields_ = [
        ("results", ctypes.POINTER(Qwen3TTSResult)),
        ("num_results", ctypes.c_size_t),
    ]

# Load Library
lib_path = os.path.abspath("../build/libqwen3-tts.so")
if not os.path.exists(lib_path):
    print(f"Library not found at {lib_path}")
    sys.exit(1)

tts_lib = ctypes.CDLL(lib_path)

# Configure argument and return types
tts_lib.qwen3_tts_init.restype = ctypes.c_void_p
tts_lib.qwen3_tts_default_params.restype = Qwen3TTSParams
tts_lib.qwen3_tts_load_models.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
tts_lib.qwen3_tts_load_models.restype = ctypes.c_bool
tts_lib.qwen3_tts_get_last_error.argtypes = [ctypes.c_void_p]
tts_lib.qwen3_tts_get_last_error.restype = ctypes.c_char_p

tts_lib.qwen3_tts_synthesize_batch.argtypes = [
    ctypes.c_void_p,                       # ctx
    ctypes.POINTER(ctypes.c_char_p),       # texts
    ctypes.c_size_t,                       # num_texts
    ctypes.c_char_p,                       # reference_audio_path
    Qwen3TTSParams                         # params
]
tts_lib.qwen3_tts_synthesize_batch.restype = ctypes.POINTER(Qwen3TTSBatchResult)

tts_lib.qwen3_tts_free_batch_result.argtypes = [ctypes.POINTER(Qwen3TTSBatchResult)]
tts_lib.qwen3_tts_free.argtypes = [ctypes.c_void_p]

def main():
    print("Initializing Qwen3-TTS engine via C-API...")
    ctx = tts_lib.qwen3_tts_init()
    if not ctx:
        print("Failed to initialize TTS context")
        return

    print("Loading models (this might take a few seconds)...")
    success = tts_lib.qwen3_tts_load_models(ctx, b"../models", b"")
    if not success:
        err = tts_lib.qwen3_tts_get_last_error(ctx)
        print(f"Failed to load models: {err.decode('utf-8') if err else 'Unknown error'}")
        tts_lib.qwen3_tts_free(ctx)
        return

    print("Models loaded successfully. Synthesizing batch...")
    
    texts_list = [
        b"Checking out the new C API batching, amazing!",
        b"Segundo audio generado dinamicamente desde ctypes en Python."
    ]
    
    # Create C array of strings
    CStrArray = ctypes.c_char_p * len(texts_list)
    c_texts = CStrArray(*texts_list)
    
    params = tts_lib.qwen3_tts_default_params()
    
    # Synthesize
    batch_res = tts_lib.qwen3_tts_synthesize_batch(ctx, c_texts, len(texts_list), b"../audioref/voz_test_0493.wav", params)
    
    if batch_res:
        res_ptr = batch_res.contents
        print(f"\nBatch processing finished! Generated {res_ptr.num_results} results.")
        for i in range(res_ptr.num_results):
            result = res_ptr.results[i]
            if result.success:
                print(f" - Result {i+1}: Success! audio_len={result.audio_len} samples, rate={result.sample_rate}")
                
                # Copy float data from pointer
                float_array = (ctypes.c_float * result.audio_len).from_address(ctypes.addressof(result.audio_data.contents))
                
                # Convert float32 [-1, 1] to PCM16
                def float_to_pcm16(f):
                    f = max(-1.0, min(1.0, f))
                    return int(f * 32767.0)
                
                pcm_data = b''.join(struct.pack('<h', float_to_pcm16(f)) for f in float_array)
                
                out_path = f"../output/test_api_output_{i}.wav"
                with wave.open(out_path, "w") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(result.sample_rate)
                    wav_file.writeframes(pcm_data)
                
                print(f"   -> Saved to {out_path}")
            else:
                err = result.error_msg.decode('utf-8') if result.error_msg else "Unknown"
                print(f" - Result {i+1}: Failed: {err}")
        
        # Cleanup
        tts_lib.qwen3_tts_free_batch_result(batch_res)
    else:
        print("Synthesize batch returned null pointer!")

    tts_lib.qwen3_tts_free(ctx)
    print("TTS engine freed. Python script exiting gracefully.")

if __name__ == "__main__":
    main()
