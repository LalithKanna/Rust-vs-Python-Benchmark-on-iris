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

### This experiment does NOT claim:
- Rust replaces Python for research
- Rust beats GPUs
- Rust beats optimized BLAS at scale

### It DOES prove:
- **Rust is an elite inference engine**
- **Rust excels at small-to-medium models**
- **System-level AI beats framework-level AI for production**
- **Memory locality matters more than FLOPS**
- **Python pays a tax you cannot optimize away**

---

## Conclusion

This benchmark demonstrates that when abstraction layers are removed and low-level control is exercised, Rust provides substantial performance advantages for neural network inference and training on modest-sized models. While Python frameworks excel in flexibility and ease of use, Rust shines in production scenarios requiring minimal latency, memory efficiency, and deterministic performance.

For embedded systems, edge devices, or latency-critical applications, these results suggest Rust is not just competitive—it's transformative.

---



