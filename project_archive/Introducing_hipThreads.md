Introducing hipThreads: A C++ - Style Concurrency Library for AMD GPUs
Contents
\* Blog Roadmap <#blog-roadmap>
\* Understanding hipThreads <#understanding-hipthreads>
\* How hipThreads work <#how-hipthreads-work>
\* hipThreads Prerequisites <#hipthreads-prerequisites>
\* Build and Installation <#build-and-installation>
\* How to use hipThreads in a CMake project <#how-to-use-hipthreads-in-
a-cmake-project>
\* SAXPY Sample Code <#saxpy-sample-code>
o Step 1: CPU Baseline with |std::thread| <#step-1-cpu-baseline-
with-std-thread>
o Step 2: First GPU Port - Drop-in Replacement with 1-Wide |
hip::thread| <#step-2-first-gpu-port-drop-in-replacement-with-1-
wide-hip-thread>
o Step 3: Optimizing with Multi-Fiber Execution <#step-3-
optimizing-with-multi-fiber-execution>
\* Performance Benefits <#performance-benefits>
\* Incremental Porting Summary <#incremental-porting-summary>
\* Summary <#summary>
\* Footnotes <#footnotes>
\* Disclaimers <#disclaimers>
Introducing hipThreads: A C++ - Style Concurrency Library for AMD
GPUs# <#introducing-hipthreads-a-c-style-concurrency-library-for-amd-gpus>
Introducing hipThreads: A C++ - Style Concurrency Library for AMD GPUs
February 19, 2026 by Charles Boyd, Kelvin Lui , Daniel Mcintosh , Marko Savic.
7 min read. | 1834 total words.
Software tools & optimizations
C++ , HPC , Installation , Performance
Developers , HPC

In this blog, you will learn how to accelerate C++ code developed for
the CPU to run on AMD GPUs using \*hipThreads\* by incrementally porting
familiar |std::thread| patterns to GPU-resident |hip::thread| code. We
walk through a step-by-step SAXPY example, explain key concepts like
persistent threads and fibers, and share real performance results to
help you evaluate when this model fits your workload.
hipThreads introduces a GPU execution model that lets you launch and
coordinate work using idioms you already know from the C++ Concurrency
Support Library. Instead of beginning your GPU journey by learning
kernel configuration, grid/block semantics, and ad-hoc synchronization,
you can write |hip::thread|, |hip::mutex|, |hip::lock\_guard|, and |
hip::condition\_variable| code that feels structurally similar to your
existing |std::thread|-driven CPU programs—making first contact with GPU
compute feel like an incremental extension of existing C++ expertise,
not a wholesale shift in mental model.
The design of hipThreads lets teams port CPU concurrency regions
incrementally: replace |std::thread| with |hip::thread|, adapt
synchronization where needed, and move logic onto the GPU without
immediately restructuring everything into bulk kernels. For newcomers to
GPU programming, it reduces cognitive load - developers can experiment
with parallelism using concepts they already trust, then dive deeper
into HIP specifics only as optimization demands.
In short, hipThreads aims to make AMD GPU compute more accessible, more
maintainable, and more aligned with modern C++ concurrency practices,
accelerating both learning curves and codebase evolution.
Blog Roadmap# <#blog-roadmap>
This post assumes you’re comfortable with modern C++ concepts,
especially multi-threading with |std::thread|. If you can write CPU
parallel code with |std::thread|, you already have most of what you need
for hipThreads.
You’ll encounter a few GPU-specific concepts as you work through the
examples:
\*
\*Device vs. host memory\*: GPUs have separate memory from CPUs
\*
\*Device functions\*: Mark code to run on the GPU with |\_\_device\_\_|
Don’t worry - we introduce each concept exactly when you need it in the
step-by-step SAXPY tutorial below, with clear explanations and working
code. You’ll learn these GPU fundamentals naturally as you port your
first parallel algorithm. For deeper background, see the HIP Programming
Guide .
We’ve structured this introduction around the following core sections to
provide a clear starting point for exploring hipThreads:
\*
Understanding hipThreads <#understanding-hipthreads>
\*
How hipThreads work <#how-hipthreads-work>
\*
Prerequisites <#hipthreads-prerequisites>
\*
Build and Installation <#build-and-installation>
\*
Use hipThreads <#how-to-use-hipthreads-in-a-cmake-project>
\*
Sample code <#saxpy-sample-code>
\*
Performance benefits <#performance-benefits>
\*Note\*: Our discussion here emphasizes the overall hipThreads
approach rather than detailed API usage. The synchronization
primitives mentioned (|hip::mutex|, |hip::lock\_guard|, |
hip::condition\_variable|, etc.) work similarly to their |std::|
counterparts but are designed for GPU execution. For complete API
documentation and detailed explanations of how these primitives
work, please refer to our hipThreads API Reference Documentation
.
Understanding hipThreads# <#understanding-hipthreads>
hipThreads fills a gap between two existing ecosystems. rocThrust offers
powerful host-side facilities (algorithms, memory management, STL-like
utilities) but does not provide a |std::thread|-style mechanism to start
GPU-resident threaded control flows. libhipcxx delivers device-side
building blocks (e.g. |hip::atomic| mirroring |std::atomic|) once code
is already running on the GPU, but it does not define how you initiate
structured, thread-like execution there. hipThreads bridges these worlds
by introducing |hip::thread|, a |std::thread|-inspired abstraction, plus
supporting synchronization primitives, giving developers a familiar on-
ramp from CPU concurrency to persistent, cooperative GPU execution.
The introduction of hipThreads delivers a suite of familiar C++‑style
GPU concurrency abstractions enabling dynamic parallelism without
excessive kernel launches:
\*
\*Familiar API\*: Launch GPU work with |hip::thread| much like |
std::thread|
\*
\*Less overhead\*: Persistent scheduler eliminates constant kernel
launching
\*
\*Cooperative\*: Threads can yield to another thread
\*
\*Standard primitives\*: Use familiar synchronization primitives that
mirror their CPU counterparts
How hipThreads work# <#how-hipthreads-work>
Instead of launching many separate GPU kernels, hipThreads launches one
long-running “manager” that stays on your GPU. When you create a |
hip::thread|, it gets added to a work queue, and the manager picks it up
and runs it alongside other threads.
Think of it as having a persistent worker pool on your GPU rather than
hiring and firing workers for every small task. This approach gives you
a familiar API where you can write |hip::thread| just like |
std::thread|, while eliminating the overhead of constant kernel launching.
Additionally, each |hip::thread| can run with multiple \*“fibers”\* -
parallel execution lanes that work together within a single virtual
thread, taking advantage of the GPU’s SIMD (Single Instruction, Multiple
Data) architecture. This allows you to distribute work efficiently
across the GPU’s vector units while maintaining the familiar threading
abstraction. For example, when processing an image, different fibers
within the same thread might handle different pixels (0, 4, 8… vs 1, 5,
9…), automatically spreading the workload across the GPU’s parallel
processing capabilities without requiring you to manually manage the
low-level details.
This fiber-based execution model is conceptually similar to CPU
vectorization technologies like AVX-512. While an AVX-512 instruction
processes 16 single-precision data elements in parallel, a |hip::thread|
running on an AMD GPU typically uses a width of 32 fibers, offering
significantly wider data parallelism within the same programming
abstraction. This means code that already benefits from CPU SIMD can
often achieve even greater speedups when ported to |hip::thread| with
minimal structural changes.
hipThreads Prerequisites# <#hipthreads-prerequisites>
hipThreads has the following prerequisites:
\*
Linux OS (Ubuntu 22.04+ is recommended)
\*
CMake 3.21+
\*
Build tools (e.g., |make| or |ninja|)
\*
\*ROCm 7.0.2\* (HIP runtime and hipcc) — \*hipThreads currently does
not work with other ROCm versions.\* See the hipThreads Prerequisites
 for detailed step-by-step ROCm 7.0.2
installation instructions.
\*
libhipcxx v2.7
o
Checkout the |release/2.7.x| branch
o
Build with |-DCMAKE\_INSTALL\_PREFIX=/opt/rocm|
o
Use |-DLIBCUDACXX\_ENABLE\_LIBCUDACXX\_TESTS=OFF| to skip tests
\*Note\*: The sample code in this blog post uses rocThrust 4.2.0
.
Install rocThrust separately by following the rocThrust installation
guide .
Build and Installation# <#build-and-installation>
\*
By default hipThreads installs under |/opt/rocm| (matching other
ROCm components). You can override this by adding |-
DCMAKE\_INSTALL\_PREFIX=| to the CMake configure command.
\*
Installing to |/opt/rocm| usually requires |sudo|.
git clone https://github.com/ROCm/hipThreads.git
cd hipThreads
mkdir build && cd build
cmake ..
make -j
sudo make install
Copy to clipboard
How to use hipThreads in a CMake project# <#how-to-use-hipthreads-
in-a-cmake-project>
To use hipThreads in your own project, add the following lines to your |
CMakeLists.txt| file:
find\_package(hipthreads REQUIRED)
[...]
target\_link\_libraries( hipthreads::hipthreads)
Copy to clipboard
SAXPY Sample Code# <#saxpy-sample-code>
We use the well-known SAXPY (Single-precision Alpha X Plus Y) operation
to demonstrate how to incrementally port a CPU-parallel algorithm to the
GPU using |hip::thread|. SAXPY computes |y[i] = alpha \* x[i] + y[i]| for
each element in arrays |x| and |y|. This example shows the natural
progression from CPU threading to GPU execution, illustrating how |
hip::thread| provides a familiar on-ramp without requiring immediate
mastery of GPU-specific concepts.
\*hipThreads SAXPY repository\*: ROCm/hipThreads
Build and run (from project root):
cd examples/saxpy/step3-simdize
mkdir build && cd build
[CXX=hipcc] cmake ..
make -j
./bin/saxpy
Copy to clipboard
\*Note\*: If hipThreads is not installed in the default |/opt/rocm|,
add |-DCMAKE\_PREFIX\_PATH=/path/to/hipThreads| to your CMake
configure command.
The sample processes 268,435,456 elements (0x10000000), with each
element undergoing 512 iterations of the computation. You can adjust
these by modifying the |N| and |NUM\_ITERATIONS| constants in |main.cxx|.
Step 1: CPU Baseline with |std::thread|# <#step-1-cpu-baseline-
with-std-thread>
We start with a standard CPU implementation using |std::thread|. This
establishes our baseline and shows the familiar threading pattern we’ll
be porting from:
#include
#include
#define N 0x10000000U
#define NUM\_ITERATIONS 512
int main() {
std::vector x(N, 1.0F);
std::vector y(N, 2.0F);
const float alpha = 2.0F;
std::vector threads(std::thread::hardware\_concurrency());
for (unsigned int i = 0; i < threads.size(); ++i) {
size\_t chunk\_size = (i < N % threads.size()) ? (N / threads.size() + 1) : (N / threads.size());
size\_t offset = (i < N % threads.size()) ? (i \* chunk\_size) : (i \* chunk\_size + N % threads.size());
threads[i] = std::thread(
[] (uint32\_t n, float a, const float \*x, float \*y) {
for (uint32\_t i = 0; i < n; ++i) {
float t = x[i];
#pragma clang loop unroll(full)
for (int j = 0; j < NUM\_ITERATIONS; ++j) {
t = a \* t + y[i];
}
y[i] = t;
}
},
chunk\_size, alpha, x.data() + offset, y.data() + offset);
}
for (auto &t : threads) {
t.join();
}
return 0;
}
Copy to clipboard
This CPU version divides the work across available CPU cores, with each
thread processing its assigned chunk of the arrays sequentially.
Step 2: First GPU Port - Drop-in Replacement with 1-Wide |
hip::thread|# <#step-2-first-gpu-port-drop-in-replacement-with-1-
wide-hip-thread>
The first step in porting to GPU is a minimal change: replace |
std::thread| with |hip::thread|. At this stage, we use the default
thread width (1 fiber per thread), making |hip::thread| a near drop-in
replacement for |std::thread|. This lets you verify GPU execution works
before optimizing for GPU-specific features.
\*Key changes from Step 1\*:
\*
Line 1: Change |#include | to |#include |
\*
Lines 5-11: Handle GPU memory allocation (explained below)
\*
Line 13: Change |std::thread| to |hip::thread|
\*
Line 18: Add |\_\_device\_\_| annotation to the lambda
#include
#include
#include
#include
#define N 0x10000000U
#define NUM\_ITERATIONS 512
int main() {
// Create and initialize vectors on the host
std::vector x\_host(N, 1.0F);
std::vector y\_host(N, 2.0F);
const float alpha = 2.0F;
// Allocate device memory and copy data to GPU
thrust::unique\_ptr x = thrust::make\_unique(N);
thrust::unique\_ptr y = thrust::make\_unique(N);
thrust::copy(x\_host.begin(), x\_host.end(), x.get());
thrust::copy(y\_host.begin(), y\_host.end(), y.get());
{ // hip::thread scope starts here
std::vector threads(hip::thread::hardware\_concurrency());
for (unsigned int i = 0; i < threads.size(); ++i) {
size\_t chunk\_size = (i < N % threads.size()) ? (N / threads.size() + 1) : (N / threads.size());
size\_t offset = (i < N % threads.size()) ? (i \* chunk\_size) : (i \* chunk\_size + N % threads.size());
threads[i] = hip::thread(
[] \_\_device\_\_(uint32\_t n, float a, const float \*x, float \*y) {
for (uint32\_t i = 0; i < n; ++i) {
float t = x[i];
for (int j = 0; j < NUM\_ITERATIONS; ++j) {
t = a \* t + y[i];
}
y[i] = t;
}
},
chunk\_size, alpha, x.get\_raw() + offset, y.get\_raw() + offset);
}
for (auto &t : threads) {
t.join();
}
} // hip::thread scope ends here
// Copy results back from GPU to host if necessary
thrust::copy(y.get(), y.get() + N, y\_host.begin());
return 0;
}
Copy to clipboard
\*Understanding GPU Memory Management\*: The CPU and GPU have separate
memory spaces. We use |thrust::unique\_ptr| to allocate GPU-accessible
memory and |thrust::copy| to transfer data between host and device. The
scoped block (|{ ... }|) ensures all |hip::thread| objects are destroyed
before we copy results back, preventing deadlocks with synchronous HIP
operations (explained below).
\*Important\*: Notice the loop structure inside the lambda remains
identical to the CPU version. Each thread still processes elements
sequentially (|for (uint32\_t i = 0; i < n; ++i)|). We haven’t yet
leveraged the GPU’s SIMD capabilities - this is purely a threading model
change.
Step 3: Optimizing with Multi-Fiber Execution# <#step-3-
optimizing-with-multi-fiber-execution>
Now we unlock the GPU’s true potential by using \*wide threads\* with
multiple fibers. Instead of each thread processing elements
sequentially, we create threads with 32 fibers that process 32 elements
in parallel using the GPU’s SIMD architecture.
\*Key changes from Step 2\*:
\*
Line 28: Add |hip::thread::max\_width()| parameter to create a 32-
fiber thread
\*
Line 30: Change loop to distribute work across fibers using |
hip::this\_thread::get\_fiber\_id()| and |hip::this\_thread::get\_width()|
#include
#include
#include
#include
#define N 0x10000000U
#define NUM\_ITERATIONS 512
int main() {
// Create and initialize vectors on the host
std::vector x\_host(N, 1.0F);
std::vector y\_host(N, 2.0F);
const float alpha = 2.0F;
// Allocate device memory and copy data to GPU
thrust::unique\_ptr x = thrust::make\_unique(N);
thrust::unique\_ptr y = thrust::make\_unique(N);
thrust::copy(x\_host.begin(), x\_host.end(), x.get());
thrust::copy(y\_host.begin(), y\_host.end(), y.get());
{ // hip::thread scope starts here
std::vector threads(hip::thread::hardware\_concurrency());
for (unsigned int i = 0; i < threads.size(); ++i) {
size\_t chunk\_size = (i < N % threads.size()) ? (N / threads.size() + 1) : (N / threads.size());
size\_t offset = (i < N % threads.size()) ? (i \* chunk\_size) : (i \* chunk\_size + N % threads.size());
threads[i] = hip::thread(hip::thread::max\_width(),
[] \_\_device\_\_(uint32\_t n, float a, const float \*x, float \*y) {
for (uint32\_t i = hip::this\_thread::get\_fiber\_id(); i < n; i += hip::this\_thread::get\_width()) {
float t = x[i];
for (int j = 0; j < NUM\_ITERATIONS; ++j) {
t = a \* t + y[i];
}
y[i] = t;
}
},
chunk\_size, alpha, x.get\_raw() + offset, y.get\_raw() + offset);
}
for (auto &t : threads) {
t.join();
}
} // hip::thread scope ends here
// Copy results back from GPU to host if necessary
thrust::copy(y.get(), y.get() + N, y\_host.begin());
return 0;
}
Copy to clipboard
\*Understanding Fibers\*: Each fiber in the thread is numbered 0 to 31
(for a width of 32). By starting the loop at |
hip::this\_thread::get\_fiber\_id()| and incrementing by |
hip::this\_thread::get\_width()|, we distribute the work: fiber 0
processes elements 0, 32, 64…, fiber 1 processes 1, 33, 65…, and so on.
The GPU executes all 32 fibers in lockstep, processing 32 elements
simultaneously.
\*Scoping and Synchronization\*: The scoped block (|{ ... }|) around |
hip::thread| objects is required in this example because we’re using
rocThrust, which makes synchronous HIP API calls (like |thrust::copy|).
The persistent kernel scheduler holds the GPU context until all |
hip::thread| objects are destroyed. Without the scope, synchronous HIP
calls would wait for the GPU context that the scheduler is holding,
causing a deadlock.
If your application uses only asynchronous HIP APIs (e.g., |
hipMemcpyAsync|, |hipMallocAsync|) with streams, you don’t need this
scoping pattern - async calls don’t block waiting for the GPU context.
However, many libraries (like rocThrust) use synchronous HIP functions
internally, so the scoping pattern is a safe practice when mixing |
hip::thread| with external libraries.
Performance Benefits# <#performance-benefits>
A small baseline port already cuts frame time from \*271.88ms\* on a /
Ryzen™ 9 9900X/ (|std::thread| + AVX‑512, 16 SIMD lanes) to \*42.60ms\* on
an /AMD Radeon™ AI PRO R9700/ (|hip::thread|, 32 lanes) — about a 6.4×
speedup.^1 <#footnote-1> This gain comes largely from wider parallelism
and the persistent multi‑fiber execution model, without deep algorithmic
changes. Single run, untuned numbers; they illustrate approach’s
potential, not peak performance.
Platform
Implementation
Time (ms)
Lanes
Relative
AMD Ryzen™ 9 9900X 12-Core
|std::thread| + AVX-512
271.88
16
1.00× (baseline)
AMD Radeon™ AI PRO R9700
|hip::thread|
42.60
32
6.4× faster^1 <#footnote-1>
\*Validated Across Diverse Workloads\*: Beyond SAXPY, we’ve tested
hipThreads on additional real-world applications, consistently achieving
2.9-6.4× speedups:
Example
Workload Type
CPU Performance
GPU Performance
Speedup
\*Sparse Matrix Multiply\*
Complex data structures
188.57s
52.30s
\*3.6×\*^3 <#footnote-3>
\*InOneWeekend Ray Tracer\*
Graphics rendering
1.610s
0.559s
\*2.9×\*^2 <#footnote-2>
These results demonstrate that hipThreads delivers meaningful
performance gains across different algorithm types—from complex data
structures to graphics and AI workloads—all with minimal code changes
and familiar C++ patterns.
\*Note\*: All benchmarks were conducted on the same test system: AMD
Ryzen™ 9 9900X, AMD Radeon™ AI PRO R9700, 64GB DDR5-4800 RAM, ASUS
TUF GAMING B850-PLUS WIFI motherboard, Ubuntu 24.04.2 LTS, using
ROCm 7.0.2 and AMDGPU driver 6.16.6.
Incremental Porting Summary# <#incremental-porting-summary>
The three-step progression demonstrates hipThreads’ design philosophy:
1.
\*Step 1 (CPU Baseline)\*: Standard |std::thread| implementation
2.
\*Step 2 (Drop-in Replacement)\*: Minimal changes to move to GPU with
1-wide threads
3.
\*Step 3 (Optimized)\*: Leverage GPU SIMD with multi-fiber execution
\*Total code changes from Step 1 to Step 3\*: 16 lines
\*
\*Added\*: 11 lines (4 includes, 4 memory allocation/transfer, 2
scoping, 1 result copy)
\*
\*Modified\*: 5 lines (namespace, device annotation, fiber width,
fiber-aware loop, pointer access)
This incremental approach lets developers:
\*
Verify GPU execution works before optimizing
\*
Understand each concept (memory management, then fibers) separately
\*
Maintain familiar threading semantics throughout
\*
Achieve significant speedups with minimal code changes
The 96% code similarity between CPU and optimized GPU implementations
shows that |hip::thread| successfully abstracts GPU complexity, making
high-performance computing accessible with a minimal learning curve.
Summary# <#summary>
In this blog, you learned how to accelerate C++ code on AMD GPUs using
\*hipThreads\* by incrementally porting familiar |std::thread| patterns to
GPU-resident |hip::thread| code. Here’s what we covered:
\*
\*Core concepts\*: How hipThreads bridges CPU and GPU programming
through persistent threads and multi-fiber execution
\*
\*Step-by-step porting\*: A three-stage progression from CPU baseline
to optimized GPU code with only 16 lines changed
\*
\*Performance gains\*: Real benchmarks showing 2.9-6.4× speedups
across diverse workloads (SAXPY, ray tracing, sparse matrix multiply)
\*
\*Practical patterns\*: Memory management with |thrust::unique\_ptr|,
and fiber-aware loop structures
hipThreads lowers the barrier to GPU programming by letting you write |
hip::thread| just like |std::thread|, while a persistent scheduler and
multi-fiber execution model handle the GPU-specific complexity behind
the scenes. Whether you’re new to GPU programming or looking for a more
maintainable way to structure GPU code, hipThreads offers an incremental
path from CPU concurrency to high-performance GPU execution.
Get the most out of hipThreads by exploring the \*hipThreads API
Reference Documentation \*, which covers the complete API,
real-world usage examples, and detailed explanations of every
synchronization primitive.
As hipThreads evolve, planned enhancements include tighter async API
integration, expanded synchronization constructs, and utilities to
simplify common porting patterns. Community feedback is encouraged:
share use cases, feature requests, pain points, or integration concerns
to help refine priorities and guide upcoming releases.
Stay tuned for future updates, deeper performance tuning guides, and
additional samples.
Footnotes# <#footnotes>
/[1] Testing by AMD as of February 2026 on the AMD Radeon™ AI PRO R9700
using ROCm 7.0.2 and AMDGPU driver 6.16.6, with HIP Threads running on
the GPU versus standard threads on the CPU, on a test system configured
with an AMD Ryzen™ 9 9900X, AMD Radeon™ AI PRO R9700, 64GB DDR5-4800
RAM, ASUS TUF GAMING B850-PLUS WIFI motherboard, and Ubuntu 24.04.2 LTS,
using the SAXPY (Single-precision A times X plus Y) computation function
test. System manufacturers may vary configurations, yielding different
results. RPS-167./
/[2] Testing by AMD as of February 2026 on the AMD Radeon™ AI PRO R9700
using ROCm 7.0.2 and AMDGPU driver 6.16.6, with HIP Threads running on
the GPU versus standard threads on the CPU, on a test system configured
with an AMD Ryzen™ 9 9900X, AMD Radeon™ AI PRO R9700, 64GB DDR5-4800
RAM, ASUS TUF GAMING B850-PLUS WIFI motherboard, and Ubuntu 24.04.2 LTS,
using the “Ray Tracing in One Weekend” ray traced rendering test. System
manufacturers may vary configurations, yielding different results. RPS-168./
/[3] Testing by AMD as of February 2026 on the AMD Radeon™ AI PRO R9700
using ROCm 7.0.2 and AMDGPU driver 6.16.6, with HIP Threads running on
the GPU versus standard threads on the CPU, on a test system configured
with an AMD Ryzen™ 9 9900X, AMD Radeon™ AI PRO R9700, 64GB DDR5-4800
RAM, ASUS TUF GAMING B850-PLUS WIFI motherboard, and Ubuntu 24.04.2 LTS,
using the Sparse Matrix Multiply (pwtk.mtx) test. System manufacturers
may vary configurations, yielding different results. RPS-169./
Disclaimers# <#disclaimers>
Third-party content is licensed to you directly by the third party that
owns the content and is not licensed to you by AMD. ALL LINKED THIRD-
PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF
SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO
CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU
ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY
ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

Contents
\* Blog Roadmap <#blog-roadmap>
\* Understanding hipThreads <#understanding-hipthreads>
\* How hipThreads work <#how-hipthreads-work>
\* hipThreads Prerequisites <#hipthreads-prerequisites>
\* Build and Installation <#build-and-installation>
\* How to use hipThreads in a CMake project <#how-to-use-hipthreads-in-
a-cmake-project>
\* SAXPY Sample Code <#saxpy-sample-code>
o Step 1: CPU Baseline with |std::thread| <#step-1-cpu-baseline-
with-std-thread>
o Step 2: First GPU Port - Drop-in Replacement with 1-Wide |
hip::thread| <#step-2-first-gpu-port-drop-in-replacement-with-1-
wide-hip-thread>
o Step 3: Optimizing with Multi-Fiber Execution <#step-3-
optimizing-with-multi-fiber-execution>
\* Performance Benefits <#performance-benefits>
\* Incremental Porting Summary <#incremental-porting-summary>
\* Summary <#summary>
\* Footnotes <#footnotes>
\* Disclaimers <#disclaimers>
\* Terms and Conditions
\* Privacy
\* Trademarks
\* Supply Chain Transparency
\* Fair and Open Competition
\* UK Tax Strategy
\* Cookie Policy
\* Cookie Settings <#cookie-settings>
