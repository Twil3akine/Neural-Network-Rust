#![allow(dead_code)]
#![allow(deprecated)]



use core::f64;

use rand::{Rng};



pub fn linear(x: f64) -> f64 {
    x
}

pub fn relu(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { x }
}

pub fn sigmoid(x: f64) -> f64 {
    (1.0) / (1.0 + (-x).exp())
}

pub fn softmax(xs: &[f64]) -> Vec<f64> {
    let max = xs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let exps: Vec<f64> = xs
        .iter()
        .map(|x| (x-max).exp())
        .collect::<Vec<f64>>();

    let sum: f64 = exps
        .iter()
        .sum::<f64>();

    exps.iter().map(|x| x/sum).collect::<Vec<f64>>()
}

/*
 * 微分
 */
pub fn linear_deriv(_x: f64) -> f64 {
    1.0
}

pub fn relu_deriv(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { 1.0 }
}

pub fn sigmoid_deriv(x: f64) -> f64 {
    let s: f64 = sigmoid(x);
    s * (1.0 - s)
}



pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    fn composition(&self, input: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(input)
            .map(|(w, x)| w*x)
            .sum::<f64>()
        + self.bias
    }

    pub fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64>
            = (0..input_size)
            .map(|_| rng.gen_range(-0.05..0.05))
            .collect();
        let bias: f64 = rng.gen_range(-0.05..0.05);

        Self {
            weights,
            bias,
        }
    }

    pub fn forward(&self, input: &[f64]) -> f64 {
        let z: f64 = self.composition(input);

        sigmoid(z)
    }
}



pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(input_size: usize, layer_size: usize) -> Self {
        let neurons: Vec<Neuron> = (0..layer_size)
            .map(|_| Neuron::new(input_size))
            .collect();

        Self {
            neurons
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(input))
            .collect()
    }

    pub fn forward_with_softmax(&self, input: &[f64]) -> Vec<f64> {
        let raw_output: Vec<f64> = self
            .neurons
            .iter()
            .map(|neuron| neuron.composition(input))
            .collect::<Vec<f64>>();

        softmax(&raw_output)
    }
}



pub struct Model {
    pub layers: Vec<Layer>,
}

impl Model {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        
        let hidden1: Layer = Layer::new(input_size, 32);
        let output: Layer = Layer::new(32, output_size);

        Self {
            layers: vec![
                hidden1,
                output
            ],
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut out = input.to_vec();

        for (idx, layer) in self.layers.iter().enumerate() {
            if idx == self.layers.len()-1 {
                out = layer.forward_with_softmax(&out);
            } else {
                out = layer.forward(&out);
            }
        }

        out
    }

    pub fn train(&mut self, input: &[f64], target: &[f64], lr: f64) -> f64 {
        let mut activations: Vec<Vec<f64>> = vec![input.to_vec()];
        let mut zs: Vec<Vec<f64>> = vec![];
        let mut out = input.to_vec();

        for (idx, layer) in self.layers.iter().enumerate() {
            let z: Vec<f64> = if idx == self.layers.len()-1 {
                layer.neurons.iter()
                    .map(|n| n.composition(&out))
                    .collect()
            } else {
                layer.neurons.iter()
                    .map(|n| n.composition(&out))
                    .collect()
            };
        
            zs.push(z.clone());
        
            out = if idx == self.layers.len()-1 {
                softmax(&z)
            } else {
                z.iter().map(|&v| sigmoid(v)).collect()
            };
        
            activations.push(out.clone());
        }

        let mut delta: Vec<f64> = activations
            .last()
            .unwrap()
            .iter()
            .zip(target.iter())
            .map(|(o, t)| o-t)
            .collect::<Vec<f64>>();

        let num_layers = self.layers.len();

        for idx in (0..num_layers).rev() {
            let layer = &mut self.layers[idx];
            let z = &zs[idx];
            let a = &activations[idx];

            let grad: Vec<f64> = if idx == num_layers - 1 {
                delta.clone()
            } else {
                delta
                    .iter()
                    .zip(z.iter())
                    .map(|(d, z)| d * sigmoid_deriv(*z))
                    .collect::<Vec<f64>>()
            };

            for (jdx, neuron) in layer.neurons.iter_mut().enumerate() {
                for kdx in 0..neuron.weights.len() {
                    neuron.weights[kdx] -= lr * grad[jdx] * a[kdx];
                }
                neuron.bias -= lr * grad[jdx];
            }

            if idx > 0 {
                let prev_layer_size = self.layers[idx-1].neurons.len();
                let mut new_delta = vec![0.0; prev_layer_size];

                for jdx in 0..new_delta.len() {
                    for (kdx, neuron) in self.layers[idx].neurons.iter().enumerate() {
                        new_delta[jdx] += grad[kdx] * neuron.weights[jdx];
                    }
                }
                delta = new_delta;
            }
        }

        activations
            .last()
            .unwrap()
            .iter()
            .zip(target.iter())
            .map(|(o, t)| {
                let o_safe = o.max(1e-10);
                -(t * (o_safe).ln())
            })
            .sum::<f64>()
        / target.len() as f64
    }
}
