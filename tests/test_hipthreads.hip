#include <hip/hip_runtime.h>
#include <hip/thread>
#include <vector>
#include <thread>
#include <iostream>
#include <chrono>
#include <cmath>

#define N 0x1000000U // ~16 Million elements for testing throughput
#define NUM_ITERATIONS 10

int main() {
    std::cout << "Starting hipThreads Isolation Test (SAXPY on pure HIP VRAM)\n";
    
    // 1. Setup Host Data
    std::vector<float> x_host(N, 1.0F);
    std::vector<float> y_host(N, 2.0F);
    const float alpha = 3.0F; // Expected: 3.0 * 1.0 + 2.0 = 5.0
    
    // 2. Allocate VRAM buffers manually
    float *d_x, *d_y;
    hipError_t err1 = hipMalloc((void**)&d_x, N * sizeof(float));
    hipError_t err2 = hipMalloc((void**)&d_y, N * sizeof(float));
    
    if (err1 != hipSuccess || err2 != hipSuccess) {
        std::cerr << "Failed to allocate VRAM via hipMalloc.\n";
        return 1;
    }
    
    std::cout << "VRAM buffers allocated directly with hipMalloc.\n";
    
    // 3. Transfer Host to Device
    hipMemcpy(d_x, x_host.data(), N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_y, y_host.data(), N * sizeof(float), hipMemcpyHostToDevice);
    
    decltype(std::chrono::steady_clock::now()) start;
    
    std::cout << "Starting hipThreads Fiber loop...\n";
    { // CRITICAL SCOPE: Persistent scheduler lifecycle
        std::vector<hip::thread> threads(hip::thread::hardware_concurrency());
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        for (unsigned int i = 0; i < threads.size(); ++i) {
            size_t chunk_size = (i < N % threads.size()) ? (N / threads.size() + 1) : (N / threads.size());
            size_t offset =     (i < N % threads.size()) ? (i * chunk_size)         : (i * chunk_size + N % threads.size());
            
            threads[i] = hip::thread(hip::thread::max_width(),
                [] __device__(uint32_t n, float a, const float *x, float *y) {
                    for (uint32_t i = hip::this_thread::get_fiber_id(); i < n; i += hip::this_thread::get_width()) {
                        float t = x[i];
                        for (int j = 0; j < NUM_ITERATIONS; ++j) {
                            t = a * t + y[i];
                        }
                        y[i] = t;
                    }
                },
                chunk_size, alpha, d_x + offset, d_y + offset);
        }
        
        // We start the clock right before the join to exclude C++ instantiation
        start = std::chrono::steady_clock::now();
        
        // Block until all threads finish to return context to host.
        for (auto &t : threads) {
            t.join();
        }
    } // CRITICAL SCOPE END. Destructors run, releasing ROCm context avoiding Deadlock.
    
    auto finished = std::chrono::steady_clock::now();
    
    // 4. Transfer Device to Host
    hipMemcpy(y_host.data(), d_y, N * sizeof(float), hipMemcpyDeviceToHost);
    
    // 5. Verification
    float t = 1.0F;
    for (int j = 0; j < NUM_ITERATIONS; ++j) {
        t = alpha * t + 2.0F;
    }
    
    bool passed = true;
    for (uint32_t i = 0; i < N; ++i) {
        if (std::abs(y_host[i] - t) > 0.0000001F) {
            std::cerr << "Mismatch at index [" << i << "]: " << y_host[i] << " (Expected " << t << ").\n";
            passed = false;
            break; // Stop at first error
        }
    }
    
    if (passed) {
        std::chrono::nanoseconds duration = finished - start;
        std::cout << "SUCCESS! Time to run SAXPY on " << N << " elements = " << duration.count() / 1000000.0 << " ms\n";
    }
    
    // Cleanup
    hipFree(d_x);
    hipFree(d_y);
    
    return passed ? 0 : 1;
}
