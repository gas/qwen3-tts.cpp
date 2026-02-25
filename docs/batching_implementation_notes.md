# Qwen3-TTS Batch Generation Implementation Notes

This document details the technical solutions and workarounds implemented to achieve stable batch generation with ROCm (HIP) GPU acceleration in the `qwen3-tts.cpp` port.

## 1. Positional Embeddings and 3D Reshape in RoPE
The backend (`ggml_rope`) inherently requires position embeddings (`inp_pos`) to be a contiguous 1D array (`ggml_is_vector`). This presented a challenge for batched operations where different sequences have distinct token index progressions (e.g., due to left-padding).

**Solution**: 
Instead of trying to modify the low-level CUDA kernels of GGML, we flatten the batch dimension just before applying RoPE:
1. `Qcur` and `Kcur` are reshaped from 4D `[head_dim, n_heads, n_tokens, batch_size]` to 3D `[head_dim, n_heads, n_tokens * batch_size]`.
2. The `inp_pos` tensor must be strictly initialized as a 1D vector of size `n_tokens * batch_size`.
3. After `ggml_rope_ext` execution, `Qcur` and `Kcur` are reshaped back to their original 4D form.
*Important Note*: Passing uninitialized memory to `inp_pos` for dimensions beyond `n_tokens` (e.g., leaving the remaining `batch_size - 1` regions uninitialized) propagates unbounded random positional scales across the GPU, resulting in heavy "alien noise" artifacts in all batched audios beyond the first.

## 2. Prefilling Offset and the ChatML Wrapper
To construct the prompt, the tokenizer prepends the required ChatML format: `<|im_start|> assistant \n`. These tokens consume exactly the first 3 indices of the generated `text_tokens`.

**Problem**: If these tokens are fed directly into the trailing generative sequence, the vocoder will audibly recite the word "assistant" at the beginning of the generated audio.
**Solution**: 
The `build_prefill_graph` correctly extracts these 3 tokens to build the `role_embed`, which are appended *before* the speaker embedding. However, the subsequent variables must be cleanly offset to prevent text generation of the prompt:
- `first_text_embed`: Reads from `text_tokens + 3`.
- `trailing_text_proj`: Reads from `text_tokens + 4`.

This precisely aligning the physical offset allows the architecture to decode the target text without vocalizing the hidden `assistant` prompt.

## 3. KV Cache Striding and Contiguity
cuBLAS and HIP blas expect batched gemm operations to be contiguously strided. 
- Transposing `Q` tensors on the fly (`ggml_permute`) produces disjoint physical strides.
- Passing fragmented strides to `ggml_mul_mat` causes silent pointer overlapping (e.g. Batch 1 reading from Batch 0's memory) returning garbage/white noise.

**Workaround**: Ensure continuous allocation for transposed multi-head queries via `Q = ggml_cont(ctx0, Q)`. Similarly, initialize the `k_cache` physically as `[head_dim, max_ctx, n_kv_heads, batch_size]` instead of `[head_dim, n_kv_heads, max_ctx, batch_size]`, bypassing the need to apply `ggml_cont` over the massive cache during runtime reads.

## 4. IM2COL Coordinate Overflows 
Vocoders relying on convolutions (like `CausalTransConvNet`) translate audio into arrays via `im2col`.
ROCm has a hard limit for `dim3` block coordinates where `dim.y` cannot exceed `65,535`. 
For audio > 6 seconds, `im2col` grids will inherently exceed 65,535 in the height axis, hanging the GPU permanently. By swapping the usage of `.x` and `.y` blocks inside `ggml_cuda_op_im2col` (where `.x` limit is `2^31-1`), we uncap the vocoder limit, allowing unlimited audio length generation in HIP.
