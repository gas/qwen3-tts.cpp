# qwen3_tts.cpp: esta rama exp/mega-graph

Nada por ahora.
Leer el [plan de implementación](./docs/megagraph_implementation_plan.md)


---
# Rama anterior exp/ggmlop

Read the [custom ops architecture](./docs/custom_ops_architecture.md)
Read the [static graph capture strategy](./docs/static_graph_capture_strategy.md)

Qwen3-TTS In-Device Optimization

## 1. El Conflicto Original del PCIe (Fase 1 a 23)
Inicialmente, el predictor autorregresivo (Encargado de decodificar 14 frames acústicos iterativos por texto inferido) funcionaba re-inicializando los grafos GGML en *cada* paso. Recrear el grafo en CPU y enviarlo a la GPU con *ggml_backend_sched_alloc_graph* tomaba ~10ms extra por frame.

La solución inicial fue la **Captura de Grafo Estática** (Static Graph Capture), asignando un Schedule independiente y reteniendo *ggml_backend_sched_graph_compute*. 

Esta fase nos dejó chocando con un segundo muro: La predicción requería descargar un token a la CPU desde VRAM de forma bloqueante (*ggml_backend_tensor_get*) para procesarlo en C++ y volver a enviarlo a la gráfica para el siguiente paso, provocando sincronizaciones constantes (Overhead de "GPU Wait / Sync" ~ 240ms por frame global).

## 2. In-Device GPU Sampling (Fases 24 a 28)
Decidimos que la única forma de liberar a la CPU de las esperas transaccionales de bus era ejecutar el código en su totalidad en GPU, incluyendo la operación Custom Python -> GGML *autoregressive_sample*. 

- **Custom Operador C++**: Agregamos al *ggml.h* y *ggml-cuda.cu* el nuevo nodo `GGML_OP_AUTOREGRESSIVE_SAMPLE` unificado bajo nuestro multiplexor híbrido de Voice Profiling (`GGML_OP_MAP_CUSTOM3`).
- **Problema de Enlace (Layout / Reshape)**: Al enlazar el grafo asincrónicamente mediante *ggml_backend_tensor_copy_async* se descubrió un problema de Layout. El grafo Sampler compilaba a forma 3D colapsada (`[1, 1, batch_size]`) mientras nuestros grafos de capas del transformador preveían tensores unidimensionales puros (`[batch_size]`). GGML rompía con un `GGML_ASSERT` por Strides desalineadas.

## 3. El Triunfo del "Anti-patrón" (Fases 29 a 39)
Intentando rodear el fallo dimensional, diseñé un puente Host-Side Asíncrono en C++ (Un Relay de buffers RAM de CPU). El proceso ordenaba la escritura del Output 3D a la RAM y la subsiguiente lectura instantánea como 1D de nuevo a la GPU, todo encadenado al mismo Kernel Stream de CUDA. 

**Resultados inesperados y Análisis:**
Este Relay de RAM asincrónico incrementó el rendimiento a ¡6.0 FPS! 

Ante este éxito, el Usuario instó a reparar por qué el Relay no podía suprimirse mediante una copia D2D real. Editando *ggml.c* para nacer todos los tensores `pred_token` en 1D puro y saltándonos la protección `ggml_are_same_layout`, comprobamos qué pasaba al usar D2D Copy Pura en AMD RDNA3 (rx7900xtx).
- **Resultados de la Copia D2D**: El rendimiento se desplomó drásticamente a 3 FPS de nuevo (235ms code prediction delay).

**La causa raíz** quedó científicamente probada: La latencia no proviene del ancho de banda PCI-e puramente, sino de la profundidad del **Command Queue del Stream HIP/ROCm**. Al retirar el Bypass Asíncrono de C++, la CPU escupía instantáneamente todas las peticiones a la controladora gráfica empantanando el scheduler con las 14 mallas de grafos y forzando castigos severos de cache del WG Processor. El Bypass es, esencialmente, un "Micro-Throttle" o **Stream Pacer** ideal para GPU AMDs de Consumo masivo.

Esta optimización (Batching Asíncrono D2H/H2D) soporta hasta Batch=7 alcanzando el soñado _Real-Time Factor (RTF)_ de ~0.9x.


## ggml
rocm: add support for host-side stream relay via async D2H/H2D
```
- Implement efficient tensor data relay using ggml_backend_tensor_get_async 
  and ggml_backend_tensor_set_async.
- Enables Host-Side Stream Relay Bypass to minimize GPU stall during 
  Code Predictor inference steps.
- Optimized for ROCm/HIP graphs by maintaining command queue continuity 
  without explicit host synchronization.
```

## qwen3_tts.cpp
feat: implement hip_graph architecture with Stream Relay Bypass
```
- Integrated hip_graphs for the Code Predictor stage to reduce dispatch overhead.
- Architecture: Host-Side Stream Relay Bypass.
  1. D2H: ggml_backend_tensor_get_async to fixed std::vector mapping.
  2. H2D: ggml_backend_tensor_set_async from vector to next stage input.
- Result: Improved pipelining between acoustic features and code prediction.
- Support for .q3vp protocol metadata handling during relay.
```
