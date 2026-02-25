# Fixing Text Embeddings Alignment and Output Pad Blocking

## Problemáticas Diagnosticadas
1. **Verborrea [assistant](file:///home/gas/code/Qwen3-TTS/qwen_tts/inference/qwen3_tts_model.py#269-271) y Mutación de Voz (Audio 0):** En C++, el método nativo para construir el grafo LLM extraía rígidamente los 3 primeros tokens para usarlos como *tags de asstistente*, inyectando el *Speaker Embedding* inmediatamente después, y leyendo el 4to token como inicio de frase. Al remover de [TextTokenizer](file:///home/gas/code/qwen3-tts.cpp/src/text_tokenizer.cpp#42-43) las etiquetas ChatML `<|im_start|>assistant\n`, empujamos la oración de los usuarios 3 casillas hacia atrás. Por consiguiente, la voz perdía el embedding clonado al principio, y luego recobra la clonación a mitad, debido a un grave desfasaje de M-RoPE frente a los tokens reales.
2. **Artefactos y Ruido Completo en Audio 1:** Al tragarse los 3 primeros tokens, la oración más corta "Puede que sí" quedó totalmente muda de prefill textual, por ende, emitía solo ruido de Vocoder vacío (lenguaje alienígena ininteligible).
3. **Fugas de Vocoder Post-EOS:** Porque a pesar de que indicábamos `is_finished[b]`, no detuvimos la recolección de los tokens hacia el vector `output[b]`, llenando el batch más corto con `tts_pad_embed` ruido estático hasta que la oración más larga concluyese. 

## Proposed Changes

### [MODIFY] [src/text_tokenizer.cpp](file:///home/gas/code/qwen3-tts.cpp/src/text_tokenizer.cpp)
- **Revertiremos** [encode_for_tts](file:///home/gas/code/qwen3-tts.cpp/src/text_tokenizer.cpp#296-303) para volver a inyectar firmemente `<|im_start|>`, [assistant](file:///home/gas/code/Qwen3-TTS/qwen_tts/inference/qwen3_tts_model.py#269-271), y `\n`. ¡El modelo físico VRAM lo exige para su mapeo de role!
- Eliminaremos el cierre espurio `<|im_end|>\n` sobrante que forzaba el Early Exit incorrecto que detectamos antes.

### [MODIFY] [src/tts_transformer.cpp](file:///home/gas/code/qwen3-tts.cpp/src/tts_transformer.cpp)
1. Re-alinear el [text_tokens](file:///home/gas/code/qwen3-tts.cpp/src/tts_transformer.cpp#946-1012) en [build_prefill_graph](file:///home/gas/code/qwen3-tts.cpp/src/tts_transformer.cpp#1013-1159) (líneas ~1115 y ~1141) saltando correctamente los índices de encabezado `+ 3` y `+ 4`, en vez de duplicar `text_tokens[0]`. Esto resolverá la "Verborrea" permanentemente.
2. En [generate_batch](file:///home/gas/code/qwen3-tts.cpp/src/tts_transformer.cpp#2993-3249) de [tts_transformer.cpp](file:///home/gas/code/qwen3-tts.cpp/src/tts_transformer.cpp) (~3233), inyectar un cerco asíncrono para que si `is_finished[b] == true`, deje de sumar `current_step_codes[b]` al vector maestro `output[b]`.

## Verification Plan
1. Correr el test local.
2. Escuchar la oración " Hola mundo ". Ya no debe balbucear [assistant](file:///home/gas/code/Qwen3-TTS/qwen_tts/inference/qwen3_tts_model.py#269-271) (el offset estará saltándolo fonéticamente al estar en la capa de metaprompt detrás del Vocoder). La voz debe ser clonada idénticamente desde el segundo 0 al 100%. El Audio 1, al no perderse, debe sonar completamente.
