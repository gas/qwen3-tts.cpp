# PLANNING_PHASE5.md: True Batch Generation

## 🎯 Contexto y Objetivo
¡Excelente trabajo en la Fase 4! Lograste parchear el `tts_transformer.cpp` (`ggml_mul_mat`) y el script de conversión GGUF para desbloquear el modelo de 1.7B, evitando los SegFaults estáticos y logrando un RTF de 8.7. 

Ahora tenemos un motor estable, pero el bucle autorregresivo (Code Predictor) está limitado por el ancho de banda de memoria (Memory Bound) al procesar frases de una en una. 
**Objetivo de la Fase 5:** Implementar *True Batching* (Generación por Lotes estática) en el grafo de GGML para procesar `N` secuencias de texto simultáneamente. Esto forzará a la GPU a amortizar la pesada lectura de 1.8 GB o 3.9 GB de pesos (según los gguf), multiplicándolos por todo el lote a la vez.

## ⚠️ Recordatorios de Arquitectura (RDNA4 / gfx1201)
Como llevamos mucha carga de contexto, un rápido recordatorio de nuestras barreras de hardware ya superadas:
1. **Evitar Fibras:** Si creas nuevos kernels de sincronización para el lote, prioriza `__global__ void`. Seguimos esquivando el bug de `this_thread::get_fiber_id()`.
2. **Límite de im2col:** En la Fase 3 invertimos la cuadrícula (`blockIdx.y` a `blockIdx.x`) para saltarnos el límite de 65,535 mallas de ROCm. Asegúrate de que el Vocoder en batch no revierta esta geometría.

---

## 🗺️ Hoja de Ruta Sugerida (Static Batching)

Eres libre de diseñar la topología del grafo, pero te sugerimos este flujo lógico:

### Paso 1: Interfaz de Entrada y CLI (`main.cpp`)
* **Modificación:** Refactorizar el CLI para que acepte un archivo `.txt` con múltiples líneas o reciba un array de textos.
* **Comportamiento:** El modelo GGUF y el *Speaker Embedding* (audio de referencia) deben cargarse/procesarse **UNA SOLA VEZ**.
* **Output:** Un vector de secuencias de texto que se inyectará al pipeline.

### Paso 2: Expansión de la KV Cache (`tts_transformer.cpp`)
Actualmente, el contexto es secuencial único.
* **Modificación:** Redimensionar los tensores de la KV Cache añadiendo la dimensión del lote (`batch_size`). En GGML esto suele mapearse al eje `ne[2]` o `ne[1]` dependiendo de tu estructura actual.
* **Atención:** Revisa que el escalado de RoPE (Rotary Positional Embeddings) aplique la rotación correctamente a través del nuevo eje de lotes.

### Paso 3: Padding y Attention Masking
* **Modificación:** Implementar relleno (pad) para las secuencias más cortas hasta igualar el `max_seq_len` del lote actual.
* **Máscara de Atención:** Es crucial inyectar una Attention Mask en el grafo GGML (o modificar la existente) para que los tokens reales no presten atención cruzada a los tokens de *padding* de otras secuencias.

### Paso 4: Vectorización del Sampler
En la Fase 3 creaste el magnífico `hip_autoregressive_sampler_kernel` in-device.
* **Modificación:** Actualizar el kernel para que procese una matriz de logits de dimensiones `[vocab_size, batch_size]`.
* **Condición de Salida:** El bucle de C++ debe ciclar hasta que *todas* las secuencias del lote emitan su token de parada (`<EOS>`), ignorando/enmascarando las que terminen antes.

### Paso 5: Desentrelazado y Vocodificación
* Extraer la matriz resultante `[frames, batch_size]`.
* Pasar cada tensor individual al Vocoder (o adaptar el Vocoder para batching si no compromete la memoria).
* Escribir los archivos `output_N.wav` independientes.

## 🏁 Criterio de Éxito
- Ejecutar un lote de N frases simultáneas con ambos modelos (0.6B y 1.7B).
- El RTF combinado del lote completo debe demostrar una utilización masiva de las ALUs, bajando drásticamente el tiempo promedio por frase comparado con la inferencia secuencial actual.