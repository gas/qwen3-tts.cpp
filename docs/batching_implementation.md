# Batching y Paralelización por GPU en Qwen3-TTS

Este documento describe la arquitectura técnica detrás del procesamiento por lotes (batching) nativo acelerado en GPU (HIP/ROCm) implementado en el motor C++ de Qwen3-TTS.

## 1. Fundamentos de la Paralelización en C++ (GGML)
A diferencia de los scripts habituales que iteran una y otra vez (procesamiento secuencial), nuestro motor aprovecha las operaciones algebraicas subyacentes de las tarjetas gráficas para ejecutar múltiples secuencias de voz **simultáneamente**. 

Cuando se envía un array de textos (ej. 6 frases) al endpoint o CLI `generate_batch`:
1. Las frases no se procesan una tras otra. Se empaquetan en un único súper-tensor matemático.
2. La arquitectura Transformer multiplica los pesos del modelo contra **todas las frases a la vez** durante la fase de *Prefill* y *Decodificación*.
3. La tarjeta gráfica (GPU) despliega miles de hilos CUDA/HIP para computar las multiplicaciones de matrices sin aumentar el tiempo de forma lineal. Por ello, generar 1 audio toma ~22s, pero generar 6 audios toma ~27s. El hardware estaba "ocioso" en peticiones individuales.

## 2. Resolviendo los Cuellos de Botella de Memoria

Para lograr esto, superamos varios bloqueos estructurales en la VRAM:
- **Matrix Strides y Caché KV:** El espacio latente del Transformer guarda el historial de la conversación (Key-Value Cache). Descubrimos que la asignación original corrompía la memoria al mezclar audios. Reestructuramos la memoria a 4 Dimensiones estrictas: `[head_dim, max_ctx, n_kv_heads, batch]`. Esto permitió el uso seguro de kernels ultrarrápidos como `hipblasSgemmStridedBatched`, que realizan multiplicaciones masivas sin tropezar matemáticamente.
- **Enmascaramiento (Padding Masks):** Como las 6 frases tienen longitudes distintas, rellenamos las más cortas con *tokens fantasmas* y le inyectamos a la red tensores de `-INF` (infinito negativo). Así, la red neuronal "ignora" el espacio vacío y aprende exclusivamente la fonética real de cada frase.
- **Early Stopping (Sileciamiento Asíncrono):** Para no malgastar cálculo, llevamos un registro de qué frase ya imprimió el token de "Fin de Audio" (`EOS`). Si una frase termina temprano (ej. "Hola."), su flujo se anula mientras esperamos a que termine la frase más larga (ej. "Esta es una frase muy larga...").

## 3. Posibilidades y Limitaciones Actuales

### ✅ Ventajas
- **Rendimiento Exponencial:** Disminución drástica del Coste-por-Audio (de ~20+ segundos a ~4 segundos).
- **Ideal para Cargas Asimétricas:** Perfecto para sintetizar libros enteros, volcando capítulos masivos (`-f archivo.txt -b 16`) aprovechando toda la VRAM.

### ⚠️ Limitaciones Estructurales (Static Batching)
- **Overhead por Relleno (Padding):** Si tu lote de 6 frases tiene 5 frases cortitas ("Sí", "No", "Hola") pero 1 frase inmensa de un párrafo completo, la GPU procesará la matriz al tamaño de la frase gigante. Las 5 frases cortas esperarán bloqueadas consumiendo VRAM inútil hasta que la más larga termine.
- **Cuello de Botella "Out-of-Memory" (OOM):** El límite del Batch (ej. 16 o 32) depende estrictamente de tu VRAM física (24GB, etc.). Si el lote más largo desborda la VRAM de golpe, la aplicación o servidor se cae.
- **Sincronización de Red:** En la API REST actual (`std::mutex`), las peticiones foráneas se apilan. Es decir, hasta que no se desocupa la GPU por completo del lote actual, el lote de solicitudes que llegó 1 segundo después no cruzará la puerta.

---

## 4. El Futuro: Dynamic Batching e In-Flight Batching

Los servidores punteros de Large Language Models (LLMs) solucionan estos problemas con **Continuous / In-Flight Batching** (implementado comúnmente en herramientas como vLLM o TGI con esquemas de memoria como *PagedAttention*).

### ¿Qué es?
En lugar de procesar lotes cerrados, el motor es una cinta transportadora abierta.
1. La frase corta ("Hola") termina en el frame 20.
2. Inmediatamente el sistema **escupé la respuesta a la red HTTP** para ese usuario.
3. El planificador (Scheduler) inyecta *al vuelo* una nueva petición HTTP que estaba en cola ocupando el hoyo de memoria libre dejado por el "Hola", sin interrumpir el cálculo continuo de la frase gigante que sigue computándose.

### Barrera Técnica en C/C++ GGML
Implementar In-Flight Batching requiere reescribir la memoria a muy bajo nivel:
- En lugar de destinar un gran bloque contiguo de VRAM estático a un "Lote 1", la VRAM se divide en "Páginas" libres dinámicas no contiguas.
- El servidor requiere un sistema circulatorio con colas Asíncronas (async/await no bloqueantes real) que alimente al bucle interno de tensores de C++ *entre* la generación de código acústico Frame a Frame, y no antes de empezar.
- Es realizable y el siguiente peldaño lógico para convertir este motor base en un gigante empresarial.
