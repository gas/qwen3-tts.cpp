# Static Graph Capture Strategy for Qwen3-TTS (`qwen3-tts.cpp`)

## Executive Summary
This document outlines the engineering process, architectural discoveries, and critical safety guidelines established while migrating the autoregressive generation loop (`predict_codes_autoregressive` and `forward_step`) to **Static Graph Execution** in a GGML/HIP environment.

The main motivation is avoiding dynamic graph reconstruction and CPU-GPU transfer overhead at every step of generation by pre-allocating an immutable buffer mapping logic.

## 1. Context Retention and Lazy Allocation
### Challenge
In the original implementation, the computation graph ( `gf_step` ) was built and allocated freshly every step via `ggml_backend_sched_alloc_graph(state_.sched, gf_step_dynamic)`, followed by an explicit `ggml_backend_sched_reset`.
A true persistent static graph must securely retain memory for its nodes over its lifetime without the backend reallocating buffers from under it.

### Solution
- **Lazy Allocation**: The graph `state_.gf_step_static` is only constructed upon the first autoregressive frame (`n_past > 0`). 
- **Isolated Allocation Schedule**: We allocate the graph strictly `ONCE` against a given scheduler, maintaining `state_.ctx_step_static = ctx0` deliberately out-of-scope without freeing it until model unload.
- **Bypassing the Reset Cycle**: Execution uses `ggml_backend_sched_graph_compute` directly, omitting `ggml_backend_sched_reset(state_.sched)` at the end of the loop frame.

## 2. Dynamic Patches on a Static Graph
With a static structure, operator dependencies like `n_past` must be modified directly via struct memory patches instead of parameter reassignments during graph build operations.

```cpp
// Iterate nodes efficiently and hot-swap attributes
for (int i = 0; i < state_.gf_step_static->n_nodes; i++) {
    struct ggml_tensor * node = state_.gf_step_static->nodes[i];
    if (node->op == GGML_OP_DIAG_MASK_INF) {
        node->op_params[0] = n_past; // dynamic scaling
    }
}
```

Similarly, dynamic tensors like Input POS (`inp_pos`) must be hot-loaded directly using `ggml_backend_tensor_set` atop identical byte-size layouts. 

## 3. Discovered Pathologies during Refactor
During the refactor, the Host encountered a critical segfault manifesting as the "Embedding token ID out of range", picking up `0x3C690200` (`1013514240`) as token identifiers. This was originally mistaken as kernels returning uninitialized arrays.

### 3.1 Unsynchronized Asynchronous Fetch (Race Condition)
Employing `ggml_backend_sched_graph_compute_async` without blocking stream synchronizations in `forward_step` allows the CPU host to launch memory fetches (`ggml_backend_tensor_get`) against floating point uncomputed aliased addresses previously mapping logits array buffers (due to Type Punning between float vs integer representation). 
**Fix**: Native substitution to synchronous `ggml_backend_sched_graph_compute`.

### 3.2 Context Memory Splicing 
The engine natively used a dynamic initialization buffer vector (`state_.compute_meta.data()`) universally. Because the Code Predictor generates dynamic graphs internally at each sequence frame, it intrinsically reused `state_.compute_meta`.
Since the static graph contexts held node structs residing directly inside this vector, the Code Predictor repeatedly and systematically shredded the header definitions of the Talker's static tensors.
This caused functions like `ggml_backend_tensor_set` to encounter unpredictable memory access violations targeting NULL backend structures (`tensor->buffer`).
**Fix**: Total segregation of memory arenas. We instantiated isolated initialization buffers (`state_.compute_meta_step_static.data()`) ensuring structurally guaranteed persistence.

### 3.3 Scheduler Clashing
Similar to Memory Splicing, both the secondary loop (`predict_codes`) and primary Talker loops were sharing `state_.sched`. The code predictor resetting `state_.sched` to construct its graphs wiped all memory maps pre-allocated to the static Talker.
**Fix**: Dual-allocation. An isolated scheduler `state_.code_pred_sched` now governs the secondary loops, eliminating collision surfaces entirely.

## 4. Architectural Rules for Future Expansions
- **NEVER** use generic vectors for context initialization if the graph is intended to be static. Allocate independent arena vectors for every persistent graph.
- **NEVER** invoke `ggml_backend_sched_reset` on a scheduler owning an active static context. Use strict Scheduler Multi-Plexing.
- Static graph inputs (`ggml_set_input`) must ALWAYS receive `ggml_backend_tensor_set` populations even if they are intermediate tensors patched between layers.

## 5. D2H / H2D Bypassing in Compute Loops
When bridging 14 sequential steps on a Code Predictor (Autoregressive sampling loop), synchronous reads to CPU create a PCIe bottleneck blocking the pipeline (e.g. 240ms per frame to extract every `pred_token`).

Instead of `ggml_backend_sched_synchronize` + `ggml_backend_tensor_get`, we enqueue the 14 fetches and pushes sequentially to the backend stream:
```cpp
// Step N - 1 Graph outputs
ggml_backend_tensor_get_async(state_.backend, pred_out, &all_pred_tokens[(step - 1) * batch_size], 0, ...);

// Step N Graph inputs directly from the pending Async Fetch D2H -> H2D relay
ggml_backend_tensor_set_async(sched[step], inp_code, &all_pred_tokens[(step - 1) * batch_size], 0, ...);
```
GGML's `tensor_copy_async` naturally refuses cross-graph transfers between different rank geometries (1D vs 2D Layout Mismatches from custom sampling nodes). 
By staging an intermediate Host View (array vector back-buffers), the CUDA Stream engine automatically aligns memory transfers in Device queues without halting the Host loop or evaluating Layout Layout/Strides assertions, yielding a 50% Latency optimization (6.0 FPS sustained).
