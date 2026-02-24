# PLANNING_PHASE2.md: Optimización Total en GPU (Qwen3-TTS)

## 🎯 Contexto General y Filosofía
El objetivo de esta fase es eliminar los cuellos de botella de transferencia de memoria (PCIe) y trasladar la computación pesada restante a la GPU. 
Eres un agente autónomo con acceso completo al código. **Tienes total libertad para elegir la mejor estrategia de implementación utilizando las APIs nativas de GGML y ROCm**, siempre y cuando respetes las siguientes restricciones de hardware.

## ⚠️ RESTRICCIÓN CRÍTICA DE HARDWARE (Bug RDNA4)
La GPU objetivo es una arquitectura **gfx1201 (RDNA4)**.
Hemos documentado un bug crítico en `hipThreads` con ROCm 7.0.2:
- **El Problema:** Llamar a la función de fibras `this_thread::get_fiber_id()` provoca un *Segmentation Fault* debido a colisiones de registros de 32/64 bits.
- **La Regla:** Si decides usar `hipThreads` (`hip::thread`) para orquestar flujos de control, DEBES inicializarlos con un ancho de 1 (`width=1`) y JAMÁS usar `get_fiber_id()`. 
- **Matemática Pesada:** Para procesamiento masivo de tensores, se prefiere el uso de kernels de HIP clásicos (`__global__ void`) lanzados con `hipLaunchKernelGGL` en lugar de fibras.

---

## 🗺️ Objetivos por Fases (Tú decides el "Cómo")

### Fase 1: Eliminar el Fallback a CPU en el Vocoder
- **Estado Actual:** En `src/audio_tokenizer_decoder.cpp`, el nodo `CUSTOM3` (activación Snake) está siendo asignado a la memoria RAM de la CPU por el planificador de GGML, lo que nos obliga a usar `hipMemcpy` en cada frame, causando latencia severa (Vocoder > 3400ms).
- **Tu Misión:** Encontrar la forma más limpia y "GGML-native" de forzar que ese tensor resida y se evalúe exclusivamente en la VRAM (`ROCm0`). 
- **Libertad de Acción:** Puedes modificar la configuración del backend, registrar la operación custom de otra manera, o como último recurso, parchear `ggml/src/ggml-hip.cpp` si la API pública de GGML no lo permite. 
- **Criterio de Éxito:** Eliminar `hipMemcpy` del callback `ggml_compute_forward_snake_custom3` y lograr que el Vocoder tarde < 100ms.

### Fase 2: Aceleración del AudioTokenizerEncoder
- **Estado Actual:** El Encoder toma ~6800ms ejecutándose en CPU. Utiliza bucles de convolución 1D (ECAPA-TDNN / Res2Net).
- **Tu Misión:** Mover la computación pesada del Encoder a la GPU.
- **Libertad de Acción:** Analiza `src/audio_tokenizer_encoder.cpp`. Puedes elegir entre:
  A) Reemplazar el grafo de GGML por kernels de HIP estándar.
  B) Usar `hip::thread` (respetando la regla de `width=1`) si consideras que la lógica secuencial se beneficia de un hilo persistente en GPU.
- **Nota CMake:** Si introduces código de dispositivo aquí, asegúrate de aplicar `HIP_SEPARABLE_COMPILATION ON` en `CMakeLists.txt` como hicimos con el decoder.

### Fase 3: Optimización del Bucle Autoregresivo (Code Predictor)
- **Estado Actual:** El despacho iterativo del *Code Predictor* (TTS Transformer) es responsable de una latencia de ~15 segundos. Esto se debe al overhead extremo de copiar los *logits* desde GPU a CPU en cada step para calcular el muestreo/softmax en el host y volver a enviar el nuevo embedding a la VRAM.
- **Tu Misión:** Diseñar una estrategia para mantener la inferencia y el muestreo de tokens en la GPU sin devolver el control al host de la CPU en cada sub-token.
- **Libertad de Acción:** Propón e implementa un kernel de muestreo autorregresivo HIP (`__global__`) iterativo que trabaje de forma asíncrona sobre los bloques de memoria local del scheduler GGML, o diseña una Custom Op dentro del grafo que auto-ingeste los nuevos embeddings minimizando las llamadas a la C++ API principal.

## 🏁 Criterios de Éxito Generales
1. Audio generado correctamente y sin estática.
2. Cero *Segmentation Faults* (Respetando la restricción de RDNA4).
3. Reducción sustancial del RTF (Real-Time Factor).