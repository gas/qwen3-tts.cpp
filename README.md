# qwen3_tts.cpp: Changes on this branch exp/ggmlop

Read the [Full implementation](./docs/static_graph_capture_strategy.md)

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

## Pacing de Streams Asíncronos (El Anti-Patrón D2H)

During the Autoregressive Optimization phase (Code Predictor, 14 frames per step), it was discovered that **pure Device-to-Device (D2D) copies via `ggml_backend_tensor_copy_async` natively degrade throughput by 50% on HIP/ROCm**, contradicting common GPU programming sense (from 145ms to 240ms per step).

The physical cause is **Stream Queue Depth Saturation**. If the Host loop does not explicitly halt to wait for device reads, the CPU enqueues 14 complete layers of `build_forward` Transformer models instantaneously into the AMD Command Queue. This huge queue chokes the WG Processors (WGP) cache management resulting in massive slow-downs per kernel execution.

**Architectural Solution:**
The Code Predictor employs a *Host-Side Stream Relay Bypass*. It purposely uses:
1. `ggml_backend_tensor_get_async` (D2H) to a fixed `std::vector` mapping.
2. `ggml_backend_tensor_set_async` (H2D) from that `vector` to the next step's input.

This acts as a transparent **Stream Pacer**. The Host CPU safely enqueues the D2H operation, but is implicitly slowed down by the memory controller resolving the PCIe bus pointer transfers. This natural "micro-throttle" trick prevents the CPU from flooding the GPU with 14 huge graphs simultaneously, achieving the golden 6.0 FPS mark and retaining optimal hardware RTF metrics for Batch size >= 1. Never replace this with a strict `tensor_copy_async` pipeline loop on RDNA architectures.