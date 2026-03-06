# Metodología de Profiling Geométrico (GGML + ROCm)

Con la implementación del Mega-Grafo estabilizada (RTF ~5.79, sin errores acústicos), nuestra investigación se traslada al nivel físico para estrechar los tiempos de evaluación matricial en la VRAM de AMD.

El flag compilatorio `-DQWEN3_TTS_TIMING=ON` es **insuficiente para este propósito**. Solo provee latencia a nivel Host (C++ "Wall-Clock Time") sobre las funciones envolventes. Para la recolección topológica de los nodos que estangulan la performance, usaremos técnicas geométricas.

## 1. El Sistema Interno de Evaluación (`GGML_PERF`)
GGML posee contadores de hardware enbebidos dentro de su planificador y framework back-end.

Para recabar tiempos granulares por *Compute Node*:
1. Debemos forzar el flag `ggml_backend_sched_set_eval_times(state_.sched, true);` en C++.
2. Y paralelamente habilitar `ggml_graph_print(gf);` previa evaluación para obtener el grafo compilado.
3. Esta información revelará **qué nodo GGML** toma mayor porción del pool de tiempo (v.g. Un bloque específico `ADD`, una copia asíncrona `CPY` producto de un `CONT`, o los `MUL_MAT` masivos del FFN y KQV).

## 2. ROCm Profiling Dinámico (`rocprof`)
Para mapear la saturación de los WGP (Workgroup Processors) de nuestra tarjeta gráfica y la utilización de la SDRAM, emplearemos los rastros genéricos del Hardware de AMD.

**Ejecución Terminal:**
```bash
rocprof --stats ./build/qwen3-tts-cli -m ./models -tts qwen3-tts-1.7b-f16.gguf -t "Benchmark" -o /tmp/audio.wav --max-tokens 50
```

Este comando expide un archivo `results.csv` resumiendo todos los lanzamientos atómicos (ej. `hip_mul_mat_kernel`, `hip_soft_max`, `hip_autoregressive_sampler_kernel`). 
Si el **Kernel Launch Overhead** (Gap Time) persiste como anomalía sobre el tiempo real de Computo (ALU Time), sabremos que el Graph Builder C++ aún está dejando fisuras de sincronización (Ejemplo: Transferencias de Streams cruzadas no mapeadas estáticamente).

## Evaluación del Diagnóstico Causal

Procedimiento de control:
* Una vez capturado el "Geométrico", documentaremos la métrica `Baseline_RTF`.
* Implementaremos las optimizaciones algebraicas (ver Plan de Optimización) **una por una**.
* Si el delta de mejora es irrisorio (ej: < 5ms) pero la complejidad estructural crece, el Parche será revertido para salvaguardar la simplicidad del código base.
* Un audio fallido ("gritos, alucinaciones") invalidará el Parche geométrico instantáneamente.

## 3. Resultados: Profiling Geométrico de Línea Base (`baseline_stats.csv`)

La captura de ROCm Profiling (7.2.0) sobre la síntesis de Qwen3-TTS 1.7B ha revelado un comportamiento contraintuitivo del hardware.

**Baseline RTF (C++ Nativo, 10-50 tokens):** `6.496`

| Rango | Kernel / Operación | % del Tiempo VRAM | Promedio Durance (ns) |
| --- | --- | --- | --- |
| 1 | `hip_autoregressive_sampler_kernel` | **68.46%** | 940,100 ns |
| 2 | `mul_mat_vec_f<__half, __half...>` | **10.33%** | 96,327 ns |
| 3 | `Cijk_Alik_Bljk_HSS_BH... (GEMM)` | **3.07%** | 239,299 ns |
| 4 | `Cijk_Alik_Bljk_SB_MT128... (GEMM)` | **2.57%** | 247,998 ns |
| 5 | `mul_mat_f<__half2...>` | **2.49%** | 52,027 ns |

La **Optimización 3 (Logits Slicing / Head Pruning)** asume la prioridad máxima del plan.

## 4. Post-Mortem de Pruebas Empíricas (Resultados de Fase 52)

Se procedió a instrumentar las mejoras matemáticas recabadas en el Plan de Implementación (`implementation_plan.md`) obteniendo resultados contraproducentes para el RTF global:

### A. Falacia Estructural de Logits Slicing
- **Hipótesis:** Cortar el tensor proyectado a lo largo del `code_pred_vocab_size = 151936` mitigaría el cuello de botella (68.4% de sobrecarga) del `hip_autoregressive_sampler_kernel`.
- **Realidad:** El vocabulario para el Sub-Grafo de Predicción Predictiva *no es de 151K*. Qwen3-TTS abstrae el pipeline fragmentando los vocabularios en Codebooks microscópicos (`code_pred_vocab_size = 2048`). 
- **Conclusión:** La latencia del 68% en el Sampler **no proviene del tamaño de la matriz**, sino inherentemente de la mecánica de "encolamiento" síncono-asíncrono de un ciclo *While Loop* en C++ que bloquea a ROCm 14 veces por frame. Slicing queda **Descartado**.

### B. El Castigo del Fused Attention (Flash Attention Ext)
- **Hipótesis:** Fusionar las ramas multi-head KQV (`Q*K -> Scale -> Mask -> Softmax -> V*KQ`) usando el driver propietario de memoria unificada de GGML (`ggml_flash_attn_ext`) ahorrará saltos de VRAM global.
- **Realidad:** El test en C++ compilado sobre ROCm 7.2.0 mostró un empeoramiento catastrófico (RTF sube a `6.811`). El Code Predictor saltó de `6379 ms` a la escalofriante cifra de `13866 ms`.
- **Razón Arquitectónica:** Qwen3-TTS procesa Autoregresión Causal **Token-a-Token ($T=1$)**. Flash Attention está diseñado para asimilar *masivos* bloques paralelos del Prefill Context. Ante $T=1$, el sobrecosto de inicializar el Kernel HIP Customizado de GGML estrangula a la ALU aplastando cualquier ganancia del Memory Catching, haciendo que la topología rudimentaria (separar `mul_mat` y `soft_max`) sea ~50% **Más Rápida** en GPUs AMD. Fused Attention queda **Revertido**.

### C. El Triunfo del "Strided Warp-Shuffle Reduction" (Fase 53)
- **Hipótesis:** El `hip_autoregressive_sampler_kernel` ocupaba el 68.46% de los nanosegundos en HIP porque la API GGML original estaba implementada mediante lógica secuencial `O(N)` forzando a un solo Hilo (Thread 0) a recorrer un *Insertion Sort* de iteraciones redundantes. Paralelizarlo por Árbol erradicaría la carga.
- **Realidad de Profiling (Final):** La latencia acumulada del Subsistema de Autoregresión Causal colapsó en caída libre. Los `14 Steps` del Code Predictor que duraban **~14,036 ms** (Línea base re-testeada) pasaron abruptamente a **3,135 ms** (-77%). El throughput escaló un +350%.
- **Resultado Macro:** El sistema superó holgadamente nuestro objetivo estricto de Tiempo Real (`RTF < 4.0`), rompiendo la barrera hacia un **RTF End-To-End de `2.675`** (Generar 50 frames tardó 4900ms).
- **Justificación de Hardware:** Al aprovechar las `Warp-level Primitives` (`__shfl_down` en bloques de 32/64 lanes RDNA4) y Memory Zancadas Alineadas (Coalesced Reads) en los 256 hilos de un `Workgroup`, erradicamos los cuellos de L1/L2 de acceso asimétrico, mitigando el O(N) a una síntesis $O(1)$ de Lectura y O($\log N$) de Computación Reductiva.

### D. El Colapso Entrópico de Warps en HIP (Fase 54)
- **Hipótesis:** Un batch de $N$ oraciones procesado paralelamente sobre la misma L1 Cache generará vocoder idéntico (ruidos calcados iniciales como "ts...") si el Generador Pseudo-Aleatorio se alimenta exlusivamente del Ticket Global de Reloj del SM (`clock64()`).
- **Realidad de Profiling:** Al lanzar 8 audios concurrentemente la reescritura Fase 53 colapsó la Semilla `base_seed + clock64()` debido a que los 8 bloques (uno por Batch) de ROCm se despachaban con tal grado de sincronía nanométrica que capturaban tics de GPU casi **Idénticos**. Esto arrojaba un Muestreo Multinomial espuriamente determinista entre frases distintas.
- **Micro-Arquitectura (RNG Noise):** Fue remediado perturbando la mutación estocástica del Sampler fusionando la geometría 3D del Despacho HIP: `(seed ^ clock64() ^ (blockIdx.y * 10243) ^ (blockIdx.z * 17))`. Esto garantizó que cada Batch diverge en su propio Árbol Estocástico.
