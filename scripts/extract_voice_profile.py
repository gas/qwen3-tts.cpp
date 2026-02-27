#!/usr/bin/env python3
import sys
import argparse
import struct
import numpy as np
import torch
import librosa
import os
from pathlib import Path

# Add the Qwen3-TTS path assuming it's adjacent or in the PYTHONPATH
try:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
except ImportError:
    # Fallback to local sibling directory
    script_dir = Path(__file__).parent.parent.absolute()
    qwen_dir = script_dir.parent / "Qwen3-TTS"
    sys.path.append(str(qwen_dir))
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

def extract_voice_profile(model_dir: str, audio_path: str, output_path: str):
    print(f"Loading model from: {model_dir}")
    # Load model (half precision is enough for extraction since we only use the tokenizer & spk encoder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(model_dir)
    model.model.to(device)
    
    print(f"Loading reference audio: {audio_path}")
    wav, sr = librosa.load(audio_path, sr=None)
    
    print("Normalizing audio...")
    if wav.ndim > 1:
        wav = wav.mean(axis=0) # convert to mono
    
    # 1. Extract speaker embedding (ECAPA-TDNN)
    target_sr_spk = model.model.speaker_encoder_sample_rate
    if sr != target_sr_spk:
        wav_spk = librosa.resample(y=wav.astype(np.float32), orig_sr=sr, target_sr=target_sr_spk)
    else:
        wav_spk = wav.astype(np.float32)
        
    print("Extracting speaker embedding...")
    spk_emb = model.model.extract_speaker_embedding(audio=wav_spk, sr=target_sr_spk)
    spk_emb_np = spk_emb.squeeze(0).cpu().numpy().astype(np.float32)
    print(f"Extracted speaker embedding of shape: {spk_emb_np.shape}")

    # 2. Extract discrete audio codes (Mimi encoder)
    print("Extracting discrete acoustic codes (ref_codes)...")
    enc_out = model.model.speech_tokenizer.encode([wav], sr=sr)
    # audio_codes shape handling depending on return type
    if isinstance(enc_out.audio_codes, list) or isinstance(enc_out.audio_codes, tuple):
        ref_codes = enc_out.audio_codes[0] # Take first batch item
    else:
        ref_codes = enc_out.audio_codes[0]
        
    ref_codes_np = ref_codes.cpu().numpy().astype(np.int32)
    
    # Expected shape: [num_frames, 8]  (num_quantizers = 8)
    num_frames, num_quantizers = ref_codes_np.shape
    print(f"Extracted {num_frames} frames of {num_quantizers}-codebook tokens.")
    
    # Pack to binary format
    print(f"Saving profile to {output_path}...")
    try:
        with open(output_path, "wb") as f:
            # Header: "Q3VP" (Qwen3 Voice Profile)
            f.write(b"Q3VP")
            
            # Metadata structure expected by C++:
            # uint32_t version
            # uint32_t sample_rate
            # uint32_t spk_dim
            # uint32_t num_frames
            # uint32_t num_quantizers
            f.write(struct.pack("<I", 1))       # version
            f.write(struct.pack("<I", 24000))   # sample_rate
            f.write(struct.pack("<I", spk_emb_np.shape[0])) # spk_dim
            f.write(struct.pack("<I", num_frames))
            f.write(struct.pack("<I", num_quantizers))

            # Data sections
            f.write(spk_emb_np.tobytes())
            f.write(ref_codes_np.tobytes())
            
        print(f"Successfully extracted {output_path}")
    except Exception as e:
        print(f"Error saving profile: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract offline voice profile (.q3vp) from reference audio using qwen_tts Python package.")
    parser.add_argument("-m", "--model", required=True, help="Path to the Qwen3-TTS HuggingFace model directory.")
    parser.add_argument("-i", "--input", required=True, help="Path to the reference audio (.wav).")
    parser.add_argument("-o", "--output", help="Path to save the extracted profile array (default: replaces .wav with .q3vp).")
    
    args = parser.parse_args()
    
    output_path = args.output
    if not output_path:
        output_path = os.path.splitext(args.input)[0] + ".q3vp"
        
    extract_voice_profile(args.model, args.input, output_path)
