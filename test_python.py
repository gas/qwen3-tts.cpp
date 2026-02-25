from transformers import AutoTokenizer
import torch
import sys
sys.path.append("/home/gas/code/Qwen3-TTS")

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models import Qwen3TTSProcessor

model_path = "/home/gas/code/qwen3-tts.cpp/models/Qwen3-TTS-12Hz-0.6B-Base"
processor = Qwen3TTSProcessor.from_pretrained(model_path)
model = Qwen3TTSForConditionalGeneration.from_pretrained(model_path)

# Nota: Para TTS base con procesador custom, a veces requiere specific kwargs
data = processor("Hola, esta es una prueba de voz generada con qwen tres.", return_tensors="pt")
out = model.generate(**data, max_new_tokens=300)

print(out[0])
