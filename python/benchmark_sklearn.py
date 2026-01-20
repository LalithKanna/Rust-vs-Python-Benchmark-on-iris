"""
Scikit-learn MLPClassifier benchmark
Configured to match Rust architecture as closely as possible
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
import json
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix as sklearn_cm
from typing import Dict

# Set thread count
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def load_and_preprocess_data(filepath: str, seed: int = 42):
    """Load and preprocess iris dataset"""
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Extract features and labels
    X = df.iloc[:, 1:5].values.astype(np.float64)
    
    # Map labels to integers
    label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    y = df.iloc[:, 5].map(label_map).values
    
    # Shuffle with same seed
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
    std[std == 0] = 1.0
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_test, y_train, y_test


def run_benchmark(data_path: str = 'data/iris.csv',
                  seed: int = 42,
                  batch_size: int = 32,
                  learning_rate: float = 0.01,
                  epochs: int = 1000) -> Dict:
    """Run sklearn MLPClassifier benchmark"""
    
    print("=" * 60)
    print("Scikit-learn MLPClassifier Benchmark")
    print("=" * 60)
    print(f"Seed: {seed}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print()
    
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"Threads: {os.environ.get('OMP_NUM_THREADS', 'unknown')}")
    print()
    
    # Baseline memory
    baseline_memory_mb = get_memory_usage_mb()
    
    # Data loading
    print("Loading dataset...")
    load_start = time.perf_counter()
    df = pd.read_csv(data_path)
    data_loading_time_ms = (time.perf_counter() - load_start) * 1000
    print(f"Data loaded in {data_loading_time_ms:.3f} ms\n")
    
    # Preprocessing
    preprocess_start = time.perf_counter()
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path, seed)
    preprocessing_time_ms = (time.perf_counter() - preprocess_start) * 1000
    
    print(f"Dataset shapes:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print()
    
    # Create model matching Rust architecture: 4 → 8 → 3
    # Note: sklearn uses different initialization, but we try to match settings
    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation='relu',
        solver='sgd',
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        learning_rate='constant',
        max_iter=epochs,
        random_state=seed,
        shuffle=False,  # We pre-shuffled
        verbose=False,
        early_stopping=False,
        n_iter_no_change=epochs,  # Disable early stopping
        warm_start=False,
    )
    
    # Training
    print("Training...")
    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time_secs = time.perf_counter() - training_start
    time_per_epoch_ms = (training_time_secs * 1000) / epochs
    
    print(f"Training complete in {training_time_secs:.3f} seconds!")
    print()
    
    # Peak training memory
    peak_training_memory_mb = get_memory_usage_mb()
    training_memory_used_mb = peak_training_memory_mb - baseline_memory_mb
    
    # Evaluation
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Inference timing
    inference_memory_mb = get_memory_usage_mb()
    
    inference_start = time.perf_counter()
    _ = model.predict(X_test)
    inference_time_us = (time.perf_counter() - inference_start) * 1_000_000
    
    # Confusion matrix
    cm = sklearn_cm(y_test, test_pred)
    
    # Results
    results = {
        'framework': 'sklearn',
        'version': sklearn.__version__,
        'seed': seed,
        'batch_size': batch_size,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'data_loading_time_ms': data_loading_time_ms,
        'preprocessing_time_ms': preprocessing_time_ms,
        'total_training_time_s': training_time_secs,
        'time_per_epoch_ms': time_per_epoch_ms,
        'inference_time_us': inference_time_us,
        'training_memory_mb': training_memory_used_mb,
        'inference_memory_mb': inference_memory_mb,
        'blas_backend': 'sklearn-builtin',
        'num_threads': int(os.environ.get('OMP_NUM_THREADS', 1)),
        'confusion_matrix': cm.tolist(),
        'loss_curve': model.loss_curve_,
    }
    
    # Print results
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test  Accuracy: {test_acc*100:.2f}%")
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
    print("=" * 60)
    
    return results


def save_results(results: Dict, output_dir: str = 'results'):
    """Save benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata CSV
    with open(f'{output_dir}/sklearn_benchmark_metadata.csv', 'w') as f:
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
    np.savetxt(f'{output_dir}/sklearn_loss_curve.csv',
               np.column_stack((np.arange(len(results['loss_curve'])), results['loss_curve'])),
               delimiter=',', header='epoch,loss', comments='')
    
    # Save confusion matrix
    np.savetxt(f'{output_dir}/sklearn_confusion_matrix.csv',
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