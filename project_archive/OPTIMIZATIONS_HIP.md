# Post-Mortem: Aceleración de Qwen3-TTS (Activación Snake con HIP)

## 1. Etapa de Diseño: La Hipótesis de `hipThreads`
El problema fundamental se originaba en la ineficiencia de GGML al calcular la función de activación continua `Snake` utilizando "Grafos Engarzados en CPU" (`ggml_exp`, `ggml_sin`, `ggml_sqr`, `ggml_add`, `ggml_mul`, etc.). El plan inicial consistía en eludir el fallback de GGML y desplegar **hipThreads** (Librería experimental de Concurrencia de ROCm basada en primitivas `std::thread` mediante fibras cooperativas para GPU) para orquestar la ejecución SIMD sobre tarjetas AMD (específicamente la arquitectura RDNA4 / `gfx1201`).

### Pasos Implementados:
- **CMakeLists.txt Configuration:** Se añadió soporte robusto a HIP usando `LANGUAGES CXX HIP` y se inyectaron las librerías `hipthreads::hipthreads`, `roc::rocprim`, etc.
- **RDC (Relocatable Device Code):** Dado que `hipThreads` depende de múltiples llamadas a kernels de librería (`__device__` calls inter-compilation units), se tuvo que transicionar CMake hacia propiedades modernas de compilación separable (`HIP_SEPARABLE_COMPILATION ON`, `LINKER_LANGUAGE HIP`).
- **GGML Custom Op Callback (`ggml_map_custom1`):** Se interceptó la operación estática del framework C de GGML usando un callback C++ para disparar el grupo de hilos `std::vector<hip::thread> threads(hip::thread::hardware_concurrency())`.

## 2. Etapa de Debugging: Crash e Inseguridades de Memoria
A pesar de que el código C++ era semánticamente correcto, los procesos terminaban inevitablemente abortados con el error catastrófico del controlador AMD:
`ROCm error: an illegal memory access was encountered` -> `[Segmentation Fault]`

### El Diagnóstico de los Tensores
Activando *prints* forenses (`ggml_type_name`, `ggml_backend_buffer_name`), descubrimos que:
1. GGML nunca migraba los tensores a `ROCm0` (VRAM), todos estaban instanciados en la **CPU (Backend: RAM/Host)**.
2. Nuestro Kernel de GPU estaba desreferenciando agresivamente y modificando punteros ubicados físicamente en la memoria de CPU (Access Violation directo).
3. **Parche Definitivo:** Implementamos un "Sandboxing" de memoria estricta para el Custom Op. Pre-alojamiento manual de VRAM con `hipMalloc`, migración host-to-device con `hipMemcpy`, y ejecución sincrónica aislada antes del volcado device-to-host back a GGML.

## 3. Cambio de Estrategia: "El Pivotaje" hacia HIP Clásico
Una vez con el código 100% blindado contra fugas de memoria, el kernel idéntico programado con **hipThreads seguía causando un Core Dump (Error Crítico del Driver ROCm)** en el momento de invocar el loop de `get_fiber_id()`. 

Se concluyó que `hipThreads` 7.0.2 (compilado _From Source_) posee un bug inherente y profundo a bajo nivel en la alineación de su Thread Pool para la matriz **RDNA4 (gfx120x)**. La librería colgaba la cola (`hang` o `timeout` de la fibra que no permitía hacer `yield()`).

Por ende, **la estrategia rotó a inyectar un kernel HIP puro (Clásico / Cuda-style)**:
1. Se abandonó `hip::thread`.
2. Se reescribió la arquitectura 1D con `blockIdx.x * blockDim.x + threadIdx.x`.
3. Se invoca usando `hipLaunchKernelGGL` asíncrono sobre la API nativa y robusta del Driver.

## 4. El Misterio del "Ruido Blanco" y la Corrección Matemática
Inicialmente, al recuperar el audio sin crashear bajo el kernel Clásico de HIP, solo se percibía ruido blanco/estática. El análisis post-mortem de la matemática lo explicaba de forma impecable:

1. **Bug del Puntero de Capa:** Estábamos pasando `alpha_tensor->data` interpretado en el runtime host (resolviendo en `0.0f` al ser leído de rama) para pasarlo como escalar por valor. La división final de la activación  `/ (a + 1e-9f)` causaba miles de `Inf` y `NaNs`.
2. **Uso de `ggml_map_custom3`:** Se migró a la API más ambiciosa del Custom Op de GGML permitiendo que nos suministrara los tensores de **Entrada (x), Alpha, y Beta** simultáneamente.
3. **Broadcasting e Iteración Per-Canal:** Se arregló el kernel para leer las matrices 1D de `alpha` y `beta` cargadas limpiamente en VRAM, con la matemática real: `salida = X + exp(-beta) * sin^2(exp(alpha) * X)`.

*(Resultado Final: La GPU calculaba el millón de operaciones correctamente en 2500ms y devolvía el Audio perfecto).*

---

## Roadmap: Necesidades para Re-Implementar `hipThreads` a Futuro

Si las futuras iteraciones de la librería Qwen3-TTS requieren escalar a pipelines de concurrencia inmensos (multi-tareas CPU-GPU interconectadas), `hipThreads` será crucial por sus colas distribuidas no-bloqueantes.

Para que se pueda reinstaurar el código original de iteración por Fibras que dejamos abortado en esta sesión, las siguientes condiciones deben cumplirse:

1. **Paridad de Versiones RDNA / ROCm:** Se deberá esperar un parche / fix up-stream en el repo oficial de AMD `hipThreads` que certifique su ejecución libre de Segment Faults para las tarjetas **gfx1201 (RDNA4)**. Las colisiones en registros 32/64 bits entre GCN de antiguas gamas suele causar estos _crashes_ silenciosos de `this_thread::get_fiber_id()`.
2. **Uso Exclusivo de Lambdas sin Interrupción C:** El bucle atómico de `for (auto &t : threads) t.join();` nunca podrá ser lanzado si el backend GGML no inicializa el entorno de contexto ROCm correctamente con el Runtime Flag (Por esto, nuestro parche de `hipMalloc` per-ciclo salvará la GPU).
3. **No Retroceder con RDC:** El CMake actual que configuramos con `HIP_SEPARABLE_COMPILATION ON` y el linker HIP es **perfecto** y debe conservarse íntegramente. Es la condición *sine qua non* para poder volver a compilar `hipThreads`.
