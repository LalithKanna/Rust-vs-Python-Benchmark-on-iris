# Rust vs NumPy vs Scikit-Learn: A Ground-Up Neural Network Benchmark

## Abstract

This experiment benchmarks three implementations of the same neural network:

1. **Pure Rust** (no BLAS, no external math libraries)
2. **Pure NumPy**
3. **Scikit-learn MLPClassifier**

All implementations were designed to be architecturally identical, trained on the same dataset, with identical hyperparameters, single-threaded execution, and deterministic seeds.

**The goal was not to prove that Rust is "faster in general"**, but to answer a much sharper question:

> What happens when we remove abstraction overhead and measure raw execution, memory behavior, and data movement?

**The results are decisive.**

---

## 1. Experimental Setup

### Dataset

- **Iris dataset** (150 samples)
- 4 continuous features
- 3 classes
- CSV input
- Train/Test split: 80/20
- Feature normalization using training statistics

### Neural Network Architecture (Identical Across All Frameworks)

```
Input:  4 features
Hidden: 8 neurons, ReLU
Output: 3 neurons, Softmax
```

### Training Configuration

| Parameter      | Value |
|---------------|-------|
| Optimizer     | SGD   |
| Learning Rate | 0.01  |
| Batch Size    | 32    |
| Epochs        | 1000  |
| Seed          | 42    |
| Threads       | 1     |

**No GPU. No multiprocessing. No BLAS acceleration.**

---

## 2. Code Verification (Fairness Guarantee)

### Rust Implementation

- Pure Rust math
- No BLAS, no MKL, no OpenBLAS
- Uses `ndarray` only for contiguous memory layout
- Manual forward and backward passes
- Explicit ReLU, Softmax, Cross-Entropy
- Deterministic batching
- Zero heap churn inside hot loops

### NumPy Implementation

- Pure NumPy arrays
- Same initialization ranges
- Same forward/backward logic
- Same batch slicing
- Threads explicitly pinned to 1

### Scikit-learn

- `MLPClassifier`
- Closest possible configuration
- SGD solver
- ReLU activation
- Same batch size and epochs
- Internal abstractions not bypassed (by design)

**Conclusion:** This is a legitimate apples-to-apples comparison, not a synthetic benchmark.

---

## 3. Raw Benchmark Results

### Accuracy

| Framework     | Train   | Test    |
|--------------|---------|---------|
| Rust         | 97.50%  | 96.67%  |
| NumPy        | 96.67%  | 96.67%  |
| scikit-learn | 98.33%  | 100.00% |

**Note:** Accuracy differences are statistically irrelevant for Iris. All models converge correctly. Rust has marginally better train accuracy than NumPy. Scikit-learn achieves better accuracy in both test and train due to internal optimizations.

### Training Time

| Framework     | Total Training Time |
|--------------|---------------------|
| Rust         | **0.042 s**         |
| NumPy        | 0.401 s             |
| scikit-learn | 0.454 s             |

**Rust is ~10× faster than NumPy and scikit-learn.**

### Inference Time

| Framework     | Inference Time |
|--------------|----------------|
| Rust         | **2 μs**       |
| NumPy        | 47 μs          |
| scikit-learn | 110 μs         |

Rust inference is:
- **23× faster than NumPy**
- **55× faster than scikit-learn**

### Memory Usage

| Framework     | Training Memory | Inference Memory |
|--------------|-----------------|------------------|
| Rust         | **0.57 MB**     | **19.78 MB**     |
| NumPy        | 1.65 MB         | 57.78 MB         |
| scikit-learn | 1.58 MB         | 112.06 MB        |

Rust uses:
- **~3× less training memory**
- **~6× less inference memory vs NumPy**
- **~11× less inference memory vs scikit-learn**

### Data Loading

| Framework     | Load Time   |
|--------------|-------------|
| Rust         | **0.34 ms** |
| NumPy        | 3.60 ms     |
| scikit-learn | 2.82 ms     |

**Rust is 8–10× faster just loading the CSV.**

---

## 4. Why Rust Dominates (Beyond "Rust Is Fast")

This result is not accidental. It follows directly from system-level properties.

### 4.1 Zero Abstraction Penalty

**In Rust:**
- No Python interpreter
- No dynamic dispatch
- No object boxing
- No reference counting
- No runtime shape checks

Every loop is:
- Monomorphized
- Inlined
- Bounds-checked once (or elided)
- Vectorized by LLVM

**In NumPy / scikit-learn:**
- Python → C boundary crossings
- Runtime dispatch
- Hidden memory allocations
- Internal safety checks per call

### 4.2 Memory Layout Control

**My Rust code ensures:**
- Contiguous `Array2<f64>` storage
- No temporary tensors in hot paths
- Explicit reuse of buffers
- Stack-allocated scalars

**Python frameworks:**
- Allocate intermediate arrays frequently
- Trigger garbage collection
- Inflate RSS even for small workloads

This explains why inference memory explodes in Python despite the tiny model.

### 4.3 Fused Operations

**I explicitly fused:**
- Matrix multiply + bias add
- ReLU in-place
- Softmax row-wise normalization
- Gradient computation with reused buffers

**NumPy:**
- Executes these as separate kernels
- Materializes intermediate arrays
- Pays overhead per operation

### 4.4 Deterministic Execution

**Rust execution is:**
- Single binary
- Single memory space
- No runtime variability

**Python:**
- Imports dominate startup
- Runtime state is global
- Interpreter overhead never disappears

This explains data loading and preprocessing gaps.

### 4.5 No BLAS in Rust vs NumPy (with BLAS)

For this project we have only **67 parameters**, so Rust without BLAS dominates NumPy. However, when parameters increase, Python with BLAS will dominate Rust without BLAS in speed. 

We can also implement BLAS in Rust, but it's harder on Windows—it's ideal for Linux-based systems. At that point, Rust will dominate NumPy in all cases.

---

## 5. Why scikit-learn Is Slow (By Design)

Scikit-learn optimizes for:
- API consistency
- Safety
- Flexibility
- Generality

It does **not** optimize for:
- Minimal memory
- Minimal latency
- Embedded inference
- System-level deployment

That is why it:
- Allocates aggressively
- Tracks extra state
- Stores loss history
- Uses Python object graphs

Its higher accuracy here is incidental, not architectural.

---

## 6. What This Experiment Actually Proves

This experiment demonstrates that Rust is exceptionally well suited for building machine learning products where execution speed, memory efficiency, and system-level control are primary concerns.

### 6.1 Rust as a System-Level ML Implementation Language

From this benchmark, it is clear that Rust provides:

- Deterministic performance

- Explicit control over memory allocation

- Fine-grained control over mathematical operations

- Predictable latency with low runtime overhead

- Unlike Python-based ecosystems, where most numerical computation is delegated to predefined BLAS backends hidden behind high-level APIs, Rust allows the developer to decide how computation is performed at every level.
---

This is particularly valuable when implementing:

- New or experimental AI architectures

- Custom loss functions

- Non-standard training or inference pipelines

- Memory-constrained or latency-critical systems

In Rust, the developer owns the full execution path from data loading to matrix operations. This makes it possible to optimize both time and memory intentionally, rather than relying on opaque framework behavior.
--- 

### 6.2 Flexibility in Math Backends and Hardware Targeting

A key insight from this experiment is that Rust does not lock the developer into a single numerical backend.

Depending on project requirements, Rust allows:

- Pure Rust math for minimal overhead and small to medium models for large models it can be used but will be complex to handle.

- OpenBLAS for hardware-independent deployments

- Intel MKL for maximum performance on Intel CPUs

- The ability to switch or combine backends without rewriting application logic

In contrast, Python users typically inherit whatever BLAS backend NumPy or the runtime environment provides, with limited visibility into how it is used internally.

This flexibility is critical for production systems that must run across different CPU architectures, cloud environments, edge devices, and embedded systems.
---

### 6.3 Operation Fusion, SIMD, and Memory Locality

Beyond batching and BLAS usage, Rust enables operation fusion at the language level.

In this experiment, several optimizations were applied that are difficult or impractical to express cleanly in Python:

- Fusing matrix multiplication, bias addition, and activation into a single execution path

- Performing ReLU and softmax in place

- Eliminating intermediate tensor allocations

- Maintaining contiguous memory layouts

- Allowing the compiler to auto-vectorize tight loops using SIMD

These optimizations significantly reduce instruction count, cache misses, memory bandwidth usage, and allocation overhead.

Python frameworks often execute these steps as separate kernels, materializing intermediate arrays and incurring additional overhead at each stage.
---

### 6.4 CPU Performance Without GPUs

An important outcome of this experiment is that well-designed Rust implementations can achieve excellent CPU performance even without GPUs.

By combining batching, vectorization, optional BLAS acceleration, loop fusion, and cache-friendly memory layouts, Rust can deliver high throughput and low latency on CPUs.

This makes Rust suitable for CPU-only production environments, cost-sensitive deployments, and real-time inference services. This performance profile can later be extended to GPUs or other accelerators, but strong baseline CPU performance already exists.
---

### 6.5 Training vs Deployment: Where Rust Fits Best

While Rust excels at performance and control, this experiment also highlights an important trade-off.

Rust is not ideal for rapid experimentation or highly iterative model design, where architectures change frequently and developer velocity is critical. Python remains superior for research, prototyping, exploratory modeling, and frequent architectural changes.

However, Rust is exceptionally well suited for production deployment, inference engines, latency-critical services, and memory-constrained systems.

A practical and effective workflow is:

- Train and experiment in Python

- Export the trained model

- Deploy the model in Rust for inference

This approach combines Python’s flexibility with Rust’s performance and reliability.
---

### 6.6 Final Conclusion

This experiment does not claim that Rust replaces Python for all machine learning tasks.

What it clearly demonstrates is that:

- Rust provides system-level control that Python cannot

- Rust enables performance and memory optimizations beyond framework abstractions

- Rust is an excellent choice for production machine learning systems, especially inference
---

### In summary:

Python is ideal for discovering models.
Rust is ideal for running them efficiently at scale.

This benchmark validates Rust as a practical foundation for high-performance AI systems, not as a research replacement, but as a production-grade execution engine.

---



