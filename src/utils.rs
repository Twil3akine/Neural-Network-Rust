#![allow(dead_code)]
#![allow(deprecated)]

use rand::Rng;

pub fn linear(x: f64) -> f64 {
    x
}

pub fn relu(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { x }
}

pub fn step(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { 1.0 }
}

pub fn sigmoid(x: f64) -> f64 {
    (1.0) / (1.0 + (-x).exp())
}

/*
 * sigmoid の微分
 */
pub fn sigmoid_deriv(x: f64) -> f64 {
    let s: f64 = sigmoid(x);
    s * (1.0 - s)
}

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    fn composition(&self, inputs: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(inputs)
            .map(|(w, x)| w*x)
            .sum::<f64>()
        + self.bias
    }

    pub fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64>
            = (0..input_size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let bias: f64 = rng.gen_range(-1.0..1.0);

        Self {
            weights,
            bias,
        }
    }

    pub fn forward(&self, inputs: &[f64]) -> f64 {
        let z: f64 = self.composition(inputs);

        sigmoid(z)
    }

    pub fn train(&mut self, inputs: &[f64], target: f64, lr: f64) -> () {
        let z: f64 = self.composition(inputs);
        let output: f64 = sigmoid(z);
        let error: f64 = output - target;
        let delta = error * sigmoid_deriv(z);

        for (idx, weight) in self.weights.iter_mut().enumerate() {
            *weight -= lr * delta * inputs[idx];
        }

        self.bias -= lr * delta;
    }
}
