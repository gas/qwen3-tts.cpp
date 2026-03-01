#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define HIP_CHECK(expr)                                                 \
    do {                                                                \
        hipError_t err = (expr);                                        \
        if (err != hipSuccess) {                                        \
            std::cerr << "HIP Error: " << hipGetErrorString(err)        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                    \
        }                                                               \
    } while (0)

// Micro-kernel to simulate workload
__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 1 << 20; // ~1 Million elements
    size_t size = n * sizeof(float);
    float a = 2.0f;

    float *x_h = (float*)malloc(size);
    float *y_h_standard = (float*)malloc(size);
    float *y_h_graph = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        x_h[i] = 1.0f;
        y_h_standard[i] = 2.0f;
        y_h_graph[i] = 2.0f;
    }

    float *x_d, *y_d;
    HIP_CHECK(hipMalloc(&x_d, size));
    HIP_CHECK(hipMalloc(&y_d, size));
    
    HIP_CHECK(hipMemcpy(x_d, x_h, size, hipMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // --- 1. Standard Kernel Launch (Iterative) ---
    HIP_CHECK(hipMemcpy(y_d, y_h_standard, size, hipMemcpyHostToDevice));
    
    int iterations = 1000;
    
    hipDeviceSynchronize();
    auto start_std = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(saxpy_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream, n, a, x_d, y_d);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    
    auto end_std = std::chrono::high_resolution_clock::now();
    double time_std = std::chrono::duration<double, std::milli>(end_std - start_std).count();
    
    HIP_CHECK(hipMemcpy(y_h_standard, y_d, size, hipMemcpyDeviceToHost));
    
    std::cout << "Standard " << iterations << " iterations took: " << time_std << " ms" << std::endl;

    // --- 2. HIP Graph Stream Capture ---
    HIP_CHECK(hipMemcpy(y_d, y_h_graph, size, hipMemcpyHostToDevice));
    hipDeviceSynchronize();

    hipGraph_t graph;
    hipGraphExec_t instance;

    // Begin Capture
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(saxpy_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream, n, a, x_d, y_d);
    }
    
    // End Capture
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    
    // Instantiate Graph
    HIP_CHECK(hipGraphInstantiate(&instance, graph, NULL, NULL, 0));

    auto start_graph = std::chrono::high_resolution_clock::now();
    
    // Launch Graph Mono-block
    HIP_CHECK(hipGraphLaunch(instance, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    
    auto end_graph = std::chrono::high_resolution_clock::now();
    double time_graph = std::chrono::duration<double, std::milli>(end_graph - start_graph).count();
    
    HIP_CHECK(hipMemcpy(y_h_graph, y_d, size, hipMemcpyDeviceToHost));
    
    std::cout << "Graph Launch Monolithic took: " << time_graph << " ms" << std::endl;
    std::cout << "Speedup: " << time_std / time_graph << "x" << std::endl;

    // Validar resultados
    bool match = true;
    for (int i = 0; i < n; i++) {
        if (y_h_standard[i] != y_h_graph[i]) {
            match = false;
            break;
        }
    }
    std::cout << "Results match: " << (match ? "YES" : "NO") << std::endl;

    HIP_CHECK(hipGraphExecDestroy(instance));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(x_d));
    HIP_CHECK(hipFree(y_d));
    free(x_h);
    free(y_h_standard);
    free(y_h_graph);

    return 0;
}
