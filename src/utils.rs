use std::fs::File;
use std::io::Write;

pub fn save_confusion_matrix(matrix: &Vec<Vec<usize>>, path: &str) {
    let mut file = File::create(path).unwrap();

    for row in matrix {
        let line = row
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        writeln!(file, "{}", line).unwrap();
    }
}

pub fn save_curve(values: &Vec<f64>, path: &str) {
    let mut file = File::create(path).unwrap();
    for (epoch, val) in values.iter().enumerate() {
        writeln!(file, "{},{}", epoch, val).unwrap();
    }
}

pub fn save_benchmark_metadata(
    seed: u64,
    batch_size: usize,
    train_acc: f64,
    test_acc: f64,
    total_time_secs: f64,
    time_per_epoch_ms: f64,
    inference_time_us: f64,
    data_loading_time_ms: f64,        // NEW
    preprocessing_time_ms: f64,       // NEW
    training_memory_mb: f64,          // NEW
    inference_memory_mb: f64,         // NEW
    blas_backend: &str,               // NEW
    blas_threads: usize,              // NEW
) {
    let mut file = File::create("benchmark_metadata.csv").unwrap();
    writeln!(file, "Metric,Value").unwrap();
    writeln!(file, "Seed,{}", seed).unwrap();
    writeln!(file, "BatchSize,{}", batch_size).unwrap();
    writeln!(file, "TrainAccuracy,{:.6}", train_acc).unwrap();
    writeln!(file, "TestAccuracy,{:.6}", test_acc).unwrap();
    
    // Timing metrics
    writeln!(file, "DataLoadingTime(ms),{:.6}", data_loading_time_ms).unwrap();
    writeln!(file, "PreprocessingTime(ms),{:.6}", preprocessing_time_ms).unwrap();
    writeln!(file, "TotalTrainingTime(s),{:.6}", total_time_secs).unwrap();
    writeln!(file, "TimePerEpoch(ms),{:.6}", time_per_epoch_ms).unwrap();
    writeln!(file, "InferenceTime(us),{:.6}", inference_time_us).unwrap();
    
    // Memory metrics
    writeln!(file, "TrainingMemoryUsed(MB),{:.6}", training_memory_mb).unwrap();
    writeln!(file, "InferenceMemory(MB),{:.6}", inference_memory_mb).unwrap();
    
    // System configuration
    writeln!(file, "BLASBackend,{}", blas_backend).unwrap();
    writeln!(file, "BLASThreads,{}", blas_threads).unwrap();
    writeln!(file, "BuildMode,release").unwrap();
}
