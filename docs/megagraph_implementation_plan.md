# Plan de Unificación Autoregresiva: El Mega-Grafo (Batch Predictor)

## 1. El Objetivo (Latencia < 10ms por Code Predictor Frame)
La investigación física del GPU profiling probó que los 145ms estructurales de latencia que sufrimos en ROCm RDNA no están dados por ancho de banda VRAM, sino por el **Kernel Launch Overhead** (Latency Driver).  
Lanzar 14 veces a la GPU un grafo iterativo (con sus [build_forward](./ggml/src/ggml.c#6823-6841), capas [ggml_mul_mat](./ggml/src/ggml.c#3202-3218), y [ggml_autoregressive_sample](./ggml/src/ggml.c#2452-2473)), bloquea asimétricamente el Host C++ de su Device (`hipStreamWaitEvent`). 

Para resolverlo al límite físico del silicio, crearemos un **Unrolled Graph (Mega-Grafo)**. 
Un solo [ggml_backend_sched_graph_compute_async](./ggml/src/ggml-backend.cpp#1797-1811) conectará las **392 capas (14 pasos x 28 transformers)** de principio a fin, dejando el Decoding íntegramente alojado en HIP sin devolver el control al Host C++ hasta ensamblar la respuesta completa de acústica.

## 2. Cambios Arquitectónicos Core (C++)

### A. Dimensión Estática de `n_past`
El parámetro clave que obligaba a relanzar el grafo iterativamente era el `n_past`. En un grafo unificado, la codificación posicional incrementa intrínsecamente.
- **Solución:** Reemplazaremos los parches dinámicos CPU (`positions_async_step[step] = n_past`) por una instanciación algorítmica estática. `inp_pos` será un vector `[14, batch_size]` mapeando de antemano la suma posicional para todos los frames relativos. Los nodos RoPE y Cross-Attention consumirán offsets pre-determinados, eliminando transferencias VRAM mutables.

### B. Encadenamiento Funcional (Graph Linking)
El archivo [src/tts_transformer.cpp](./src/tts_transformer.cpp) sufrirá una reestructuración de [build_code_pred_step_graph](./src/tts_transformer.cpp#1906-2082):
1.  **Iteración Cero (Prefill Externo o Input Residual):** Se recibe `inp_code` originario.
2.  **El Loop Interno de GGML:** Introduciremos un bucle `for (int step = 0; step < 14; step++)` *dentro del builder espacial de GGML*.
3.  **Forward:** Se construye la torre completa de 28 blocks de Transformer con su respectiva entrada (K/V cache incrementado por el iterador de step).
4.  **Sampling Híbrido:** El nodo terminal del step N invoca nuestro Custom Operator: `ggml_tensor * token_step = ggml_autoregressive_sample(ctx, logits, cfg);`
5.  **El Enlace Crítico (Auto-Feed):** El token emitido (`token_step` 1D) se enlaza automáticamente a la entrada del embedding del siguiente frame: `ggml_tensor * next_code = ggml_get_rows(ctx, code_embd, token_step)`.

### C. Alocación Estructural (`ggml_context`)
El `state_.ctx_code_pred_static` debe absorber `14x` veces más nodos simultáneos. Su [mem_size](./ggml/src/ggml.c#1604-1607) en la función [predict_codes_autoregressive](./src/tts_transformer.cpp#2613-2926) se reajustará calculando el tamaño de un Unrolled Graph de profundidad total para dimensionar el Session Overhead.

## 3. Modificaciones en el HIP Backend (Driver y Strides)
Alinearemos la salida final para la recolección del audio. El output final será un tensor de Rank 2 concatenado (`[batch_size, 14]`), devuelto asíncronamente con un único [tensor_get_async](./ggml/src/ggml-backend.cpp#268-281) posterior al Computation Sync, purgando todas las inter-dependencias de C++.

## 4. Riesgos Evaluados
*   **VRAM Overhead de Tensors:** El Grafo se volverá masivo en cantidad de Tensores Auxiliares. No obstante, al emplear [ggml_backend_sched_alloc_graph](./ggml/src/ggml-backend.cpp#1772-1790), el Graph Scheduler de memoria es brutalmente eficiente y reutilizará buffers para tensores de tiempo de vida corto (Compute Buffers). **No esperamos OOM.**
*   **Complejidad del KV-Cache:** El KV-cache debe apuntar a franjas iterativas secuenciales dentro de su [size](./ggml/src/ggml-backend.cpp#850-859). El builder debe calcular algebraicamente el subset de [ggml_view_1d(cache_k)](./ggml/src/ggml.c#3651-3660) para el step exacto en un Contexto Unificado. 

## 5. Plan de Ejecución 
1. Re-parametrización y despiece de [build_code_pred_step_graph](./src/tts_transformer.cpp#1906-2082) hacia un mega loop en [src/tts_transformer.cpp](./src/tts_transformer.cpp).
2. Supresión de todos los arrays intermedios (`all_pred_tokens`, `positions_async_step`) del Predictor CPU.
3. Actualización matemática de las vistas de contexto KV-Cache posicionales para soportar el encadenamiento intra-grafo en C++.
4. Compilación del Master-Graph y pruebas de rendimiento contra los 145ms logrados anteriormente.

## 6. Post-Mortem: Resolución de Alucinaciones en el Mega-Grafo (Marzo 2026)
Durante la integración inicial del Mega-Grafo, el audio emitido consistió exclusivamente en artefactos y ruido ("screams") a pesar de que el pipeline no lanzaba errores de aserción ni OOM. Tras una depuración física, se detectaron y corrigieron **dos grietas críticas en la arquitectura causal**:

### A. Fallo Estructural: Colisión Prefill vs Step 1
El bucle iterativo Causal estaba duplicando la inicialización del embedding de entrada.
- **El Error:** El código forzaba a que `step == 1` sumara analíticamente el tensor residual `inp_hidden` proyectado junto con el `codebook_0`. Sin embargo, en el diseño unificado, el Subgrafo de *Prefill* ya se engarga de procesar `codebook_0` (junto con el `hidden_state`) computando internamente $T=0$ y $T=1$. 
- **La Solución:** Extirpar `inp_hidden` del Mega-Grafo por completo. Se forzó a que el inicio del Mega-Grafo `inp_code_step1` consumiera asincrónicamente `output[b][0]` (que ahora aloja correctamente el token derivado del Prefill, $T=2$) y se normalizó el proyector de embedding para ser estricto desde el inicio del loop. Este ajuste sanó la fragmentación del `n_past` del KV Cache.

### B. Fallo Aritmético: Colapso de Dimensiones VRAM en el Sampler HIP
- **El Error:** El tensor probabilístico `logits` era envíado a `GGML_OP_AUTOREGRESSIVE_SAMPLE` perdiendo una dimensión por colapso estático (`[vocab_size, 1, batch_size]` → `[vocab_size, batch_size]`). El HIP kernel internamente utilizaba `dims == 3` sin soportar el `dims == 2`, resultando en que la variable offset `stride_batch_bytes` se leyera como basura residual. El sampler terminaba de-referenciando punteros vacíos en la VRAM para sacar al token $n_{t}$, derivando en pura entropía incontrolada.
- **La Solución:** Restauramos la indexación topológica de memoria para tensores colapsados 1D y 2D a nivel hardware en `ggml_cuda_autoregressive_sample`, recuperando la alineación determinística de los tokens. OOM controlados con `no_alloc=true` ceden libremente la gestión a ROCm.
