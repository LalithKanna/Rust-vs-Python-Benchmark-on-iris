use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use ndarray::{Array1, Array2, Axis};
#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>,
    pub b2: Vec<f64>,
}

#[derive(Debug)]
pub struct NeuralNetworkRuntime {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
}

fn random_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-0.5..0.5))
}

fn zero_vector(size: usize) -> Array1<f64> {
    Array1::zeros(size)
}

// Fused ReLU (SIMD-optimized)
#[inline]
fn relu_inplace(mut z: Array2<f64>) -> Array2<f64> {
    z.mapv_inplace(|x| if x > 0.0 { x } else { 0.0 });
    z
}

#[inline]
fn relu_derivative(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

// Fused softmax (SIMD-optimized)
#[inline]
fn softmax_inplace(mut logits: Array2<f64>) -> Array2<f64> {
    for mut row in logits.axis_iter_mut(Axis(0)) {
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|x| (x - max).exp());
        let sum: f64 = row.sum();
        row.mapv_inplace(|x| x / sum);
    }
    logits
}

// Batch cross-entropy loss
#[inline]
fn cross_entropy_loss(y_hat: &Array2<f64>, y: &Array2<f64>) -> f64 {
    let n = y_hat.nrows() as f64;
    -(&(y * &y_hat.mapv(|x| x.ln()))).sum() / n
}

impl NeuralNetworkRuntime {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        
        let input_size = 4;
        let hidden_size = 8;
        let output_size = 3;

        NeuralNetworkRuntime {
            w1: random_matrix(hidden_size, input_size, &mut rng),
            b1: zero_vector(hidden_size),
            w2: random_matrix(output_size, hidden_size, &mut rng),
            b2: zero_vector(output_size),
        }
    }

    // Batched forward pass with fusion
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // z1 = x @ W1^T + b1 (fused)
        let mut z1 = x.dot(&self.w1.t());
        z1.axis_iter_mut(Axis(0)).for_each(|mut row| {
            row += &self.b1;
        });

        // a1 = ReLU(z1) (in-place)
        let a1 = relu_inplace(z1);

        // z2 = a1 @ W2^T + b2 (fused)
        let mut z2 = a1.dot(&self.w2.t());
        z2.axis_iter_mut(Axis(0)).for_each(|mut row| {
            row += &self.b2;
        });

        // softmax(z2) (in-place)
        softmax_inplace(z2)
    }

    // Batched training step with gradient fix
    pub fn train_step(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        learning_rate: f64,
    ) -> f64 {
        let batch_size = x.nrows() as f64;

        // Forward pass (store intermediates)
        let mut z1 = x.dot(&self.w1.t());
        z1.axis_iter_mut(Axis(0)).for_each(|mut row| {
            row += &self.b1;
        });

        let a1 = relu_inplace(z1.clone());

        let mut z2 = a1.dot(&self.w2.t());
        z2.axis_iter_mut(Axis(0)).for_each(|mut row| {
            row += &self.b2;
        });

        let y_hat = softmax_inplace(z2);

        // Loss
        let loss = cross_entropy_loss(&y_hat, y);

        // Backward pass
        let delta2 = &y_hat - y; // (batch_size, 3)

        // CRITICAL FIX: Store W2 before mutation
        let w2_old = self.w2.clone();

        // Gradients for W2 and b2
        let grad_w2 = delta2.t().dot(&a1) / batch_size;
        let grad_b2 = delta2.sum_axis(Axis(0)) / batch_size;

        // Update W2 and b2
        self.w2 -= &(learning_rate * &grad_w2);
        self.b2 -= &(learning_rate * &grad_b2);

        // delta1 = (delta2 @ W2_old) * ReLU'(z1)
        let delta1 = delta2.dot(&w2_old) * &relu_derivative(&z1);

        // Gradients for W1 and b1
        let grad_w1 = delta1.t().dot(x) / batch_size;
        let grad_b1 = delta1.sum_axis(Axis(0)) / batch_size;

        // Update W1 and b1
        self.w1 -= &(learning_rate * &grad_w1);
        self.b1 -= &(learning_rate * &grad_b1);

        loss
    }

    // Batched prediction
    pub fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let probs = self.forward(x);
        Array1::from_vec(
            probs
                .axis_iter(Axis(0))
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap()
                        .0
                })
                .collect()
        )
    }

    // Export to serializable format
    pub fn to_serializable(&self) -> NeuralNetwork {
        NeuralNetwork {
            w1: self.w1.outer_iter().map(|row| row.to_vec()).collect(),
            b1: self.b1.to_vec(),
            w2: self.w2.outer_iter().map(|row| row.to_vec()).collect(),
            b2: self.b2.to_vec(),
        }
    }
}

pub fn save_model(
    nn: &NeuralNetworkRuntime,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let serializable = nn.to_serializable();
    let json = serde_json::to_string_pretty(&serializable)?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}