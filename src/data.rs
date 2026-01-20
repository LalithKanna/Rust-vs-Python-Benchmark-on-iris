use csv::ReaderBuilder;
use std::error::Error;
use rand::{SeedableRng, seq::SliceRandom};
use rand::rngs::StdRng;
use ndarray::{Array2, Array1};

#[derive(Debug, Clone)]
pub struct Sample {
    pub features: [f64; 4],
    pub label: usize,
}

fn map_label(label: &str) -> usize {
    match label {
        "Iris-setosa" => 0,
        "Iris-versicolor" => 1,
        "Iris-virginica" => 2,
        _ => panic!("Unknown class label: {}", label),
    }
}

pub fn load_iris(path: &str) -> Result<Vec<Sample>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut dataset = Vec::new();

    for record in reader.records() {
        let r = record?;

        if r.len() != 6 {
            return Err(format!("Invalid record length: {}", r.len()).into());
        }

        let features = [
            r[1].parse::<f64>()?,
            r[2].parse::<f64>()?,
            r[3].parse::<f64>()?,
            r[4].parse::<f64>()?,
        ];

        let label = map_label(&r[5]);
        dataset.push(Sample { features, label });
    }

    Ok(dataset)
}

pub fn shuffle_and_split(
    mut dataset: Vec<Sample>,
    train_ratio: f64,
    seed: u64,
) -> (Vec<Sample>, Vec<Sample>) {
    let mut rng = StdRng::seed_from_u64(seed);
    dataset.shuffle(&mut rng);
    let train_size = (dataset.len() as f64 * train_ratio).round() as usize;

    let train_set = dataset[..train_size].to_vec();
    let test_set = dataset[train_size..].to_vec();

    (train_set, test_set)
}

pub fn compute_mean_std(train_set: &Vec<Sample>) -> ([f64; 4], [f64; 4]) {
    let n = train_set.len() as f64;

    let mut mean = [0.0; 4];
    let mut std = [0.0; 4];

    for sample in train_set {
        for i in 0..4 {
            mean[i] += sample.features[i];
        }
    }

    for i in 0..4 {
        mean[i] /= n;
    }

    for sample in train_set {
        for i in 0..4 {
            let diff = sample.features[i] - mean[i];
            std[i] += diff * diff;
        }
    }

    for i in 0..4 {
        std[i] = (std[i] / n).sqrt();

        if std[i] == 0.0 {
            std[i] = 1.0;
        }
    }

    (mean, std)
}

pub fn normalize_features(
    features: [f64; 4],
    mean: &[f64; 4],
    std: &[f64; 4],
) -> [f64; 4] {
    let mut normalized = [0.0; 4];

    for i in 0..4 {
        normalized[i] = (features[i] - mean[i]) / std[i];
    }

    normalized
}

pub fn normalize_dataset(
    dataset: &Vec<Sample>,
    mean: &[f64; 4],
    std: &[f64; 4],
) -> Vec<Sample> {
    dataset
        .iter()
        .map(|sample| Sample {
            features: normalize_features(sample.features, mean, std),
            label: sample.label,
        })
        .collect()
}


// New: Convert dataset to matrix representation
pub fn dataset_to_matrices(
    dataset: &Vec<Sample>,
    num_classes: usize,
) -> (Array2<f64>, Array2<f64>) {
    let n = dataset.len();
    let mut x = Array2::<f64>::zeros((n, 4));
    let mut y = Array2::<f64>::zeros((n, num_classes));

    for (i, sample) in dataset.iter().enumerate() {
        for j in 0..4 {
            x[[i, j]] = sample.features[j];
        }
        y[[i, sample.label]] = 1.0;
    }

    (x, y)
}

// New: Get labels as array
pub fn get_labels(dataset: &Vec<Sample>) -> Array1<usize> {
    Array1::from_vec(dataset.iter().map(|s| s.label).collect())
}