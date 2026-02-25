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

-------------------------------------------------------------
[AMPLIACIÓN ESTRATÉGICA: PREPARACIÓN PARA BACKEND RESIDENTE]
Para que este esfuerzo de True Batching tenga un impacto real, el objetivo final es que el motor funcione detrás de un servidor HTTP residente.

Para lograrlo, te pido que aísles la lógica de inferencia del archivo main.cpp.

    Crea una API en C++ limpia: En lugar de que el motor lea directamente de argv, refactoriza el generador para que sea una clase o función que acepte un std::vector<std::string> inputs.

    Desacoplamiento: El motor Qwen3-TTS debe inicializarse una vez (cargando los pesos GGUF y la referencia de audio en la VRAM), quedarse en memoria, y exponer un método generate_batch(vector<string>) que devuelva un vector<vector<float>> (los audios generados).

    Al hacer esto, en un futuro cercano (Fase 6) solo tendremos que conectar cpp-httplib a esa función, acumulando las peticiones web en un buffer y pasándoselas a tu motor de lotes en ráfagas.

Prioriza primero que la matemática del Batching en GGML (Padding, Máscara de Atención y Sampler Vectorizado) funcione localmente pasándole un array estático en el código, y luego expondremos el servidor.

------------------------------
NOTA TECNICA: recuerda el alias 'torch_702' que te da acceso a las librerias rocm 7.0.2 (ya que hace 'module load rocm/7.0.2' y ya carga el entorno virtual '~/.envs/torch_rocm72/bin/activate' con torchaudio, triton, safetensors, etc específicos instalados). Si necesitas instalar algo por pip (dentro de ese entorno) puedes pedirlo.