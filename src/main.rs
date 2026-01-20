mod data;
mod model;
mod utils;
mod experiment;

use std::error::Error;
use std::time::Instant;
use sysinfo::{System, Pid};
use crate::data::load_iris;
use crate::experiment::{run_experiment, save_experiment_results};
use crate::utils::save_benchmark_metadata;

fn get_process_memory_mb() -> f64 {
    let pid = Pid::from_u32(std::process::id());
    let mut sys = System::new_all();
    sys.refresh_all();
    
    if let Some(process) = sys.process(pid) {
        process.memory() as f64 / (1024.0 * 1024.0)
    } else {
        0.0
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let blas_backend = "none"; // BLAS removed

    println!("Loading dataset...");
    let load_start = Instant::now();
    let dataset = load_iris(r"K:\Kg-intern\tuto1\irisclassificationnn\data\Iris.csv")?;
    let data_loading_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(dataset.len(), 150);
    println!("Data loaded in {:.3} ms\n", data_loading_time_ms);

    let seed = 42;
    let batch_size = 32;

    println!("Running deterministic benchmark...");
    println!("Seed: {}", seed);
    println!("Batch size: {}", batch_size);
    println!("Math backend: {}", blas_backend);
    println!();

    // Baseline memory
    let baseline_memory_mb = get_process_memory_mb();

    let result = run_experiment(&dataset, seed, batch_size);

    // Peak training memory
    let peak_training_memory_mb = get_process_memory_mb();
    let training_memory_used_mb = peak_training_memory_mb - baseline_memory_mb;

    // Inference memory
    let inference_memory_mb = get_process_memory_mb();

    save_experiment_results(&result);
    save_benchmark_metadata(
        seed,
        batch_size,
        result.train_acc,
        result.test_acc,
        result.total_time_secs,
        result.time_per_epoch_ms,
        result.inference_time_us,
        data_loading_time_ms,
        result.preprocessing_time_ms,
        training_memory_used_mb,
        inference_memory_mb,
        blas_backend,
        0,
    );

    println!("=== BENCHMARK RESULTS ===");
    println!("Train Accuracy: {:.2}%", result.train_acc * 100.0);
    println!("Test  Accuracy: {:.2}%", result.test_acc * 100.0);

    println!("\n=== TIMING ===");
    println!("Data Loading: {:.3} ms", data_loading_time_ms);
    println!("Preprocessing: {:.3} ms", result.preprocessing_time_ms);
    println!("Total Training Time: {:.3} seconds", result.total_time_secs);
    println!("Time per Epoch: {:.3} ms", result.time_per_epoch_ms);
    println!("Inference Time: {:.1} μs", result.inference_time_us);

    println!("\n=== MEMORY ===");
    println!("Baseline Memory: {:.2} MB", baseline_memory_mb);
    println!("Peak Training Memory: {:.2} MB", peak_training_memory_mb);
    println!("Training Memory Used: {:.2} MB", training_memory_used_mb);
    println!("Inference Memory: {:.2} MB", inference_memory_mb);

    println!("\n=== SYSTEM ===");
    println!("Math Backend: pure Rust (no BLAS)");

    Ok(())
}