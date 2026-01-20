use crate::data::{Sample, shuffle_and_split, compute_mean_std, normalize_dataset, dataset_to_matrices, get_labels};
use crate::model::{NeuralNetworkRuntime, save_model};
use crate::utils::{save_confusion_matrix, save_curve};
use std::time::Instant;
use ndarray::s;
use ndarray::{Array1, Array2};

pub struct ExperimentResult {
    pub train_acc: f64,
    pub test_acc: f64,
    pub losses: Vec<f64>,
    pub train_accs: Vec<f64>,
    pub test_accs: Vec<f64>,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub total_time_secs: f64,
    pub time_per_epoch_ms: f64,
    pub inference_time_us: f64,
    pub preprocessing_time_ms: f64,  // NEW
}

fn accuracy(
    nn: &NeuralNetworkRuntime,
    x: &Array2<f64>,
    labels: &Array1<usize>,
) -> f64 {
    let preds = nn.predict(x);
    let correct = preds
        .iter()
        .zip(labels.iter())
        .filter(|(p, t)| p == t)
        .count();

    correct as f64 / labels.len() as f64
}

fn confusion_matrix(
    nn: &NeuralNetworkRuntime,
    x: &Array2<f64>,
    labels: &Array1<usize>,
    num_classes: usize,
) -> Vec<Vec<usize>> {
    let preds = nn.predict(x);
    let mut matrix = vec![vec![0; num_classes]; num_classes];

    for (true_label, pred_label) in labels.iter().zip(preds.iter()) {
        matrix[*true_label][*pred_label] += 1;
    }

    matrix
}

pub fn run_experiment(dataset: &Vec<Sample>, seed: u64, batch_size: usize) -> ExperimentResult {
    // Feature 3: Start preprocessing timer
    let preprocess_start = Instant::now();
    
    let (train_set, test_set) = shuffle_and_split(dataset.clone(), 0.8, seed);

    let (mean_feat, std_feat) = compute_mean_std(&train_set);

    let train_norm = normalize_dataset(&train_set, &mean_feat, &std_feat);
    let test_norm = normalize_dataset(&test_set, &mean_feat, &std_feat);

    let (train_x, train_y) = dataset_to_matrices(&train_norm, 3);
    let (test_x, _test_y) = dataset_to_matrices(&test_norm, 3);
    
    let train_labels = get_labels(&train_norm);
    let test_labels = get_labels(&test_norm);

    // Feature 3: End preprocessing timer
    let preprocessing_time_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;

    let mut nn = NeuralNetworkRuntime::new(seed);
    let lr = 0.01;
    let epochs = 1000;

    let mut losses = Vec::new();
    let mut train_accs = Vec::new();
    let mut test_accs = Vec::new();

    let training_start = Instant::now();

    for _ in 0..epochs {
        let mut epoch_loss = 0.0;
        let n_samples = train_x.nrows();
        let mut n_batches = 0;

        // Mini-batch training
        for start in (0..n_samples).step_by(batch_size) {
            let end = (start + batch_size).min(n_samples);
            let batch_x = train_x.slice(s![start..end, ..]).to_owned();
            let batch_y = train_y.slice(s![start..end, ..]).to_owned();
            
            epoch_loss += nn.train_step(&batch_x, &batch_y, lr);
            n_batches += 1;
        }

        // Normalize loss by number of batches
        losses.push(epoch_loss / n_batches as f64);
        train_accs.push(accuracy(&nn, &train_x, &train_labels));
        test_accs.push(accuracy(&nn, &test_x, &test_labels));
    }

    let training_time = training_start.elapsed();
    let total_time_secs = training_time.as_secs_f64();
    let time_per_epoch_ms = (total_time_secs * 1000.0) / epochs as f64;

    // Measure inference time
    let inference_start = Instant::now();
    let _ = nn.predict(&test_x);
    let inference_time = inference_start.elapsed();
    let inference_time_us = inference_time.as_micros() as f64;

    save_model(&nn, "trained_model.json").unwrap();
    
    let train_acc = accuracy(&nn, &train_x, &train_labels);
    let test_acc = accuracy(&nn, &test_x, &test_labels);
    let cm = confusion_matrix(&nn, &test_x, &test_labels, 3);

    ExperimentResult {
        train_acc,
        test_acc,
        losses,
        train_accs,
        test_accs,
        confusion_matrix: cm,
        total_time_secs,
        time_per_epoch_ms,
        inference_time_us,
        preprocessing_time_ms,  // NEW
    }
}

pub fn save_experiment_results(result: &ExperimentResult) {
    save_confusion_matrix(&result.confusion_matrix, "confusion_matrix.csv");
    save_curve(&result.losses, "loss_curve.csv");
    save_curve(&result.train_accs, "train_accuracy.csv");
    save_curve(&result.test_accs, "test_accuracy.csv");
}