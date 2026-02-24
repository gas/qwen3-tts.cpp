# PLANNING.md: Optimización de Qwen3-TTS con hipThreads

## 🎯 Objetivo Principal
Reducir la latencia del **Vocoder Decoder** (actualmente representa el 27.1% o 74.9% del tiempo de ejecución, según `OPTIMIZATION.md`). Vamos a reemplazar el grafo encadenado de la función de activación **Snake** en GGML por un único kernel ejecutado en la GPU de AMD utilizando el modelo de concurrencia y "fibras" de la librería `hipThreads`.

## 📁 Archivos Objetivo
1. `CMakeLists.txt`
2. `src/audio_tokenizer_decoder.cpp` (y posiblemente renombrarlo a `.cu` o `.hip` para facilitar la compilación por parte de CMake/hipcc).

---

## 🗺️ Hoja de Ruta (Step-by-Step)

### Paso 1: Configurar CMake para hipThreads
**Agente:** Modifica el archivo `CMakeLists.txt` en la raíz.
- Busca la línea donde se incluye el paquete de threads (`find_package(Threads REQUIRED)`).
- Justo debajo, añade la instrucción para buscar hipThreads: `find_package(hipthreads REQUIRED)`.
- Ve a la declaración de la librería `audio_tokenizer_decoder` (alrededor de la línea 85).
- Cambia la configuración para asegurarte de que `src/audio_tokenizer_decoder.cpp` se compile usando `hipcc` (o configúralo como fuente HIP/CUDA).
- Añade `hipthreads::hipthreads` a `target_link_libraries(audio_tokenizer_decoder PUBLIC ...)`.

### Paso 2: Implementar la Lógica del Kernel de hipThreads
**Agente:** Modifica el archivo `src/audio_tokenizer_decoder.cpp`.
- Añade las cabeceras necesarias en la parte superior:
  ```cpp
  #include <hip/thread>
  #include <math.h>

```

* Implementa la siguiente función `__device__` pura usando la API de **fibras** (procesamiento SIMD bloque a bloque):
```cpp
__device__ void hip_snake_activation_kernel(int n, float* x, float alpha) {
    for (int i = hip::this_thread::get_fiber_id(); i < n; i += hip::this_thread::get_width()) {
        float val = x[i];
        float sin_ax = sinf(alpha * val);
        x[i] = val + (sin_ax * sin_ax) / alpha; 
    }
}

```



### Paso 3: Crear el Custom Op Callback para GGML

**Agente:** En el mismo archivo `src/audio_tokenizer_decoder.cpp`, crea el lanzador host-side.

* Define el callback para GGML que inicialice el pool persistente y llame a la función `__device__`.
* **Importante:** Usa llaves de ámbito `{ }` alrededor de `hip::thread` para asegurar que se unan (`join`) y no bloqueen el contexto de GGML.
```cpp
void ggml_compute_forward_snake_custom(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata) {
    int n_elements = ggml_nelements(dst);
    float* d_x = (float*) dst->data; 
    float alpha = *(float*)userdata;

    {
        std::vector<hip::thread> threads(hip::thread::hardware_concurrency());
        for (unsigned int i = 0; i < threads.size(); ++i) {
            threads[i] = hip::thread(hip::thread::max_width(), 
                                     hip_snake_activation_kernel, 
                                     n_elements, d_x, alpha);
        }
        for (auto &t : threads) t.join();
    }
}

```



### Paso 4: Reemplazar el Grafo en `apply_snake`

**Agente:** Localiza la función original `AudioTokenizerDecoder::apply_snake` en `src/audio_tokenizer_decoder.cpp`.

* Actualmente construye un grafo pesado usando `ggml_exp`, `ggml_reshape_3d`, `ggml_repeat`, `ggml_mul`, `ggml_sin`, `ggml_sqr` y `ggml_add`.
* **Acción:** Comenta o elimina toda la lógica interna y sustitúyela por un `ggml_map_custom1` (o su equivalente aplicable a la versión actual de GGML del repositorio).
* Utiliza la estructura `userdata` para pasar el puntero del valor `alpha` (que deberás extraer del tensor `struct ggml_tensor * alpha` asumiendo que es un escalar uniforme).
* Retorna el tensor resultante de la operación Custom.

### Paso 5: Compilación y Profiling

**Usuario/Agente:** - Compilar usando los flags de CMake instrumentados: `cmake .. -DQWEN3_TTS_TIMING=ON` seguido de `make -j`.

* Ejecutar el benchmark usando `./build/qwen3-tts-cli -m models -t "Test" -r clone.wav -o output.wav`.
* **Criterio de Éxito:** Validar en la consola que la etapa *Vocoder Decode* baje drásticamente su tiempo desde los **11,612 ms** (reportados en `OPTIMIZATION.md`) sin corromper el audio de salida.

## ⚠️ Posibles Riesgos y Consideraciones para el Agente

* **Ubicación de los Tensores:** El Custom Op en GGML asume que `dst->data` está accesible para la GPU (VRAM). GGML podría requerir asegurar que la memoria subyacente esté mapeada a HIP o transferida. Si falla la ejecución, considerar el uso de `ggml_backend_tensor_get()` para transferir si GGML está en un backend de CPU, o usar punteros nativos de dispositivo si GGML está configurado con backend CUDA/HIP.
* Si CMake no detecta `hipcc`, evalúa la posibilidad de forzar `set(CMAKE_CXX_COMPILER hipcc)` o migrar la compilación del backend de audio a `ENABLE_HIP`.
