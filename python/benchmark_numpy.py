"""
Pure NumPy implementation matching Rust architecture exactly:
- Same network: 4 → 8 (ReLU) → 3 (Softmax)
- Same initialization: uniform(-0.5, 0.5)
- Same optimizer: SGD with lr=0.01
- Same batch size: 32
- Same epochs: 1000
- Same seed: 42
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
import json
from typing import Tuple, Dict

# CRITICAL: Set thread count BEFORE importing numpy operations
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

class NeuralNetworkNumPy:
    """Pure NumPy neural network matching Rust implementation"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Network architecture: 4 → 8 → 3
        input_size = 4
        hidden_size = 8
        output_size = 3
        
        # Xavier/uniform initialization matching Rust: uniform(-0.5, 0.5)
        self.w1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
        self.b2 = np.zeros(output_size)
        
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        # Batch operation: each row is a sample
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass - batched"""
        # z1 = x @ W1.T + b1
        self.z1 = x @ self.w1.T + self.b1
        
        # a1 = ReLU(z1)
        self.a1 = self.relu(self.z1)
        
        # z2 = a1 @ W2.T + b2
        self.z2 = self.a1 @ self.w2.T + self.b2
        
        # y_hat = softmax(z2)
        self.y_hat = self.softmax(self.z2)
        
        return self.y_hat
    
    def cross_entropy_loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Cross-entropy loss averaged over batch"""
        batch_size = y.shape[0]
        # Clip to avoid log(0)
        y_hat_clipped = np.clip(y_hat, 1e-10, 1.0)
        loss = -np.sum(y * np.log(y_hat_clipped)) / batch_size
        return loss
    
    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """Backward pass with SGD update - batched"""
        batch_size = x.shape[0]
        
        # Forward pass (stores intermediates)
        y_hat = self.forward(x)
        
        # Loss
        loss = self.cross_entropy_loss(y_hat, y)
        
        # Backward pass
        # delta2 = y_hat - y
        delta2 = y_hat - y
        
        # CRITICAL: Store W2 before mutation (matching Rust fix)
        w2_old = self.w2.copy()
        
        # Gradients for W2 and b2
        grad_w2 = delta2.T @ self.a1 / batch_size
        grad_b2 = np.sum(delta2, axis=0) / batch_size
        
        # Update W2 and b2
        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2
        
        # delta1 = (delta2 @ W2_old) * ReLU'(z1)
        delta1 = (delta2 @ w2_old) * self.relu_derivative(self.z1)
        
        # Gradients for W1 and b1
        grad_w1 = delta1.T @ x / batch_size
        grad_b1 = np.sum(delta1, axis=0) / batch_size
        
        # Update W1 and b1
        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        
        return loss
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probs = self.forward(x)
        return np.argmax(probs, axis=1)


def load_and_preprocess_data(filepath: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess iris dataset matching Rust exactly"""
    
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Extract features and labels
    X = df.iloc[:, 1:5].values.astype(np.float64)
    
    # Map labels to integers
    label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    y = df.iloc[:, 5].map(label_map).values
    
    # Shuffle with same seed as Rust
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Train/test split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Normalize using training set statistics
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_test, y_train, y_test


def one_hot_encode(y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """One-hot encode labels"""
    n = len(y)
    y_encoded = np.zeros((n, num_classes))
    y_encoded[np.arange(n), y] = 1
    return y_encoded


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy"""
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Calculate confusion matrix"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark(data_path: str = 'data/iris.csv', 
                  seed: int = 42, 
                  batch_size: int = 32,
                  learning_rate: float = 0.01,
                  epochs: int = 1000) -> Dict:
    """Run complete benchmark matching Rust implementation"""
    
    print("=" * 60)
    print("NumPy Neural Network Benchmark")
    print("=" * 60)
    print(f"Seed: {seed}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"NumPy version: {np.__version__}")
    print(f"Threads: {os.environ.get('OMP_NUM_THREADS', 'unknown')}")
    print()
    
    # Baseline memory
    baseline_memory_mb = get_memory_usage_mb()
    
    # Feature 3: Data loading time
    print("Loading dataset...")
    load_start = time.perf_counter()
    df = pd.read_csv(data_path)
    data_loading_time_ms = (time.perf_counter() - load_start) * 1000
    print(f"Data loaded in {data_loading_time_ms:.3f} ms\n")
    
    # Feature 3: Preprocessing time
    preprocess_start = time.perf_counter()
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path, seed)
    y_train_onehot = one_hot_encode(y_train)
    y_test_onehot = one_hot_encode(y_test)
    preprocessing_time_ms = (time.perf_counter() - preprocess_start) * 1000
    
    print(f"Dataset shapes:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print()
    
    # Initialize model
    model = NeuralNetworkNumPy(seed=seed)
    
    # Training
    print("Training...")
    losses = []
    train_accs = []
    test_accs = []
    
    training_start = time.perf_counter()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        # Mini-batch training
        n_samples = len(X_train)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_x = X_train[start:end]
            batch_y = y_train_onehot[start:end]
            
            loss = model.backward(batch_x, batch_y, learning_rate)
            epoch_loss += loss
            n_batches += 1
        
        # Average loss over batches
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        # Calculate accuracies
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = accuracy(y_train, train_pred)
        test_acc = accuracy(y_test, test_pred)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%")
    
    training_time_secs = time.perf_counter() - training_start
    time_per_epoch_ms = (training_time_secs * 1000) / epochs
    
    print()
    print("Training complete!")
    print()
    
    # Feature 1: Peak training memory
    peak_training_memory_mb = get_memory_usage_mb()
    training_memory_used_mb = peak_training_memory_mb - baseline_memory_mb
    
    # Final evaluation
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    final_train_acc = accuracy(y_train, train_pred)
    final_test_acc = accuracy(y_test, test_pred)
    
    # Feature 1: Inference memory and time
    inference_memory_mb = get_memory_usage_mb()
    
    inference_start = time.perf_counter()
    _ = model.predict(X_test)
    inference_time_us = (time.perf_counter() - inference_start) * 1_000_000
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    
    # Results
    results = {
        'framework': 'numpy',
        'version': np.__version__,
        'seed': seed,
        'batch_size': batch_size,
        'train_accuracy': final_train_acc,
        'test_accuracy': final_test_acc,
        'data_loading_time_ms': data_loading_time_ms,
        'preprocessing_time_ms': preprocessing_time_ms,
        'total_training_time_s': training_time_secs,
        'time_per_epoch_ms': time_per_epoch_ms,
        'inference_time_us': inference_time_us,
        'training_memory_mb': training_memory_used_mb,
        'inference_memory_mb': inference_memory_mb,
        'blas_backend': 'numpy-builtin',
        'num_threads': int(os.environ.get('OMP_NUM_THREADS', 1)),
        'losses': losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'confusion_matrix': cm.tolist(),
    }
    
    # Print results
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Train Accuracy: {final_train_acc*100:.2f}%")
    print(f"Test  Accuracy: {final_test_acc*100:.2f}%")
    print()
    print("=== TIMING ===")
    print(f"Data Loading: {data_loading_time_ms:.3f} ms")
    print(f"Preprocessing: {preprocessing_time_ms:.3f} ms")
    print(f"Total Training Time: {training_time_secs:.3f} seconds")
    print(f"Time per Epoch: {time_per_epoch_ms:.3f} ms")
    print(f"Inference Time: {inference_time_us:.1f} μs")
    print()
    print("=== MEMORY ===")
    print(f"Training Memory Used: {training_memory_used_mb:.2f} MB")
    print(f"Inference Memory: {inference_memory_mb:.2f} MB")
    print()
    print("=== SYSTEM ===")
    print(f"BLAS Backend: numpy-builtin")
    print(f"Threads: {results['num_threads']}")
    print("=" * 60)
    
    return results


def save_results(results: Dict, output_dir: str = 'results'):
    """Save benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata
    metadata = {k: v for k, v in results.items() 
                if k not in ['losses', 'train_accs', 'test_accs', 'confusion_matrix']}
    
    with open(f'{output_dir}/numpy_benchmark_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save CSV format (matching Rust)
    with open(f'{output_dir}/numpy_benchmark_metadata.csv', 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"Seed,{results['seed']}\n")
        f.write(f"BatchSize,{results['batch_size']}\n")
        f.write(f"TrainAccuracy,{results['train_accuracy']:.6f}\n")
        f.write(f"TestAccuracy,{results['test_accuracy']:.6f}\n")
        f.write(f"DataLoadingTime(ms),{results['data_loading_time_ms']:.6f}\n")
        f.write(f"PreprocessingTime(ms),{results['preprocessing_time_ms']:.6f}\n")
        f.write(f"TotalTrainingTime(s),{results['total_training_time_s']:.6f}\n")
        f.write(f"TimePerEpoch(ms),{results['time_per_epoch_ms']:.6f}\n")
        f.write(f"InferenceTime(us),{results['inference_time_us']:.6f}\n")
        f.write(f"TrainingMemoryUsed(MB),{results['training_memory_mb']:.6f}\n")
        f.write(f"InferenceMemory(MB),{results['inference_memory_mb']:.6f}\n")
        f.write(f"BLASBackend,{results['blas_backend']}\n")
        f.write(f"BLASThreads,{results['num_threads']}\n")
        f.write(f"Framework,{results['framework']}\n")
        f.write(f"Version,{results['version']}\n")
    
    # Save curves
    np.savetxt(f'{output_dir}/numpy_loss_curve.csv', 
               np.column_stack((np.arange(len(results['losses'])), results['losses'])),
               delimiter=',', header='epoch,loss', comments='')
    
    np.savetxt(f'{output_dir}/numpy_train_accuracy.csv',
               np.column_stack((np.arange(len(results['train_accs'])), results['train_accs'])),
               delimiter=',', header='epoch,accuracy', comments='')
    
    np.savetxt(f'{output_dir}/numpy_test_accuracy.csv',
               np.column_stack((np.arange(len(results['test_accs'])), results['test_accs'])),
               delimiter=',', header='epoch,accuracy', comments='')
    
    # Save confusion matrix
    np.savetxt(f'{output_dir}/numpy_confusion_matrix.csv',
               results['confusion_matrix'], delimiter=',', fmt='%d')
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    results = run_benchmark(
        data_path='data/iris.csv',
        seed=42,
        batch_size=32,
        learning_rate=0.01,
        epochs=1000
    )
    
    save_results(results, output_dir='results')