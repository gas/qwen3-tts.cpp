# Custom Operations Architecture & Memory Contracts

This document establishes the fundamental architectural constraints, memory contracts, and structural rules for the `qwen3-tts.cpp` inference engine. **Any agent or developer modifying the GPU kernels, custom operations, or the GGML graph MUST read and adhere to these rules before making changes.**

## 1. The Multiplexed `MAP_CUSTOM3` Setup

Due to limitations in extending the core `ggml.h` API without upstream forks, the `GGML_OP_MAP_CUSTOM3` operation serves as a multiplexer for injecting custom CUDA/HIP kernels into the `ggml_cgraph`.

Currently, `ggml_cuda_op_map_custom3` (inside `ggml-cuda.cu`) routes execution based on the destination (`dst`) tensor type:
- **`custom_type == 1` (Snake Activation):** Used by the `AudioTokenizerDecoder` (Vocoder). Triggered by default for `GGML_TYPE_F32`. Extracts `alpha` (from `src[1]`) and `beta` (from `src[2]`) to compute $x + \frac{1}{\beta} \sin^2(\alpha x)$.
- **`custom_type == 2` (Autoregressive Sampler):** Triggered when `dst->type == GGML_TYPE_I32`. Extracts `logits` (from `src[1]`), reads dynamic `seq_len` natively, and outputs an `int32_t` token.

**Rule:** Never blindly overwrite `MAP_CUSTOM3`. Any new custom operations must be carefully multiplexed (e.g., using `dst->op_params` flags or `dst->type`) so as not to break existing Vocoder activations or Sampler logic.

## 2. Static Tensor Masking for HIP Graphs

To achieve high inference speeds, `qwen3-tts.cpp` relies on `ggml_backend_sched_graph_compute` capturing the compute graph into a static CUDA/HIP Graph (`cudaGraph_t`).

**Constraints for HIP Graphs:**
1. **Topology Cannot Change:** The order, number, and shape of nodes (`ggml_tensor`) in the graph cannot change between the Prefill (Step 0) and autoregressive decoding (Steps 1-N).
2. **Pointers Cannot Change:** The underlying memory addresses for the tensors must remain fixed. You cannot reallocate a tensor in the middle of a generation loop.
3. **Implicit Asynchronicity:** `hipGraph` operates entirely asynchronously. Injecting custom barriers like `hipDeviceSynchronize()` inside a `MAP_CUSTOM3` Node *will* deadlock the graph execution.

**How we bypass this:**
We construct the graph assuming the maximum possible sequence length and batch size for a chunk. The Context size, `pos` encodings, and `k/v_cache` sizes are statically allocated.

## 3. Autoregressive Sampler: Memory Contract

If implementing *In-Device GPU Sampling*, the `hip_autoregressive_sampler_kernel` operates under strict pointer arithmetic rules:

### A. Input Tensors & Locations
- **`logits` (Input):** A `[vocab_size, seq_len, batch_size]` tensor living entirely in GPU VRAM (HIP Device Memory). 
- **`out_token` (Output):** A `[batch_size]` `GGML_TYPE_I32` tensor living in GPU VRAM. **MUST NOT** alias `logits` (No In-Place execution).

### B. Prefill vs. Decode Stride Calculation
During Step 0 (Prefill), the context contains `N` tokens (e.g., Text Tokens + ICL Audio Codes). Thus, `logits` has a `seq_len > 1` (e.g., `ne[1] = 600`).
During Step >= 1 (Decode), `logits` has `seq_len = 1`.

The GPU kernel *must* calculate the memory offset to read only the probabilities of the *last* token dynamically:
```cpp
int seq_len = dst->src[1]->ne[1];
size_t stride_bytes = dst->src[1]->nb[1];
const float * logits = (const float *)(logits_base + (seq_len - 1) * stride_bytes);
```
**Never use `ggml_view_2d` or `ggml_reshape` to slice the logits on the CPU before passing them to the GPU.** This breaks the `nb` strides and causes the GPU to read the wrong token memory block.

## 4. Voice Cloning (`.q3vp`) and ICL Context

The architecture supports In-Context Learning (ICL) via `.q3vp` Voice Profiles.
- These profiles inject a massive block of `k_cache`/`v_cache` data (the speaker's acoustic prompt) directly into the `TTSTransformer` state at Step 0.
- Modifying the batching mechanism or the sampler kernel must account for the fact that Step 0 is processing hundreds of tokens simultaneously, relying heavily on `ggml_mul_mat` and `ggml_rope_ext` scaling.
- If the batch size expands dynamically, the `q3vp` cache injection logic will fail unless the graph was explicitly allocated to handle `max_batch_size`.
