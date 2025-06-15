#![allow(dead_code)]
#![allow(deprecated)]

use rand::{Rng};

pub enum Activation {
    Linear,
    Sigmoid,
    ReLU,
    Softmax
}

impl Activation {
    pub fn activate(&self, x: f64) -> f64 {
        match self {
            Activation::Linear => x,
            Activation::Sigmoid => (1.0) / (1.0 + (-x).exp()),
            Activation::ReLU => if x < 0.0 { 0.0 } else { x },
            Activation::Softmax => {
                panic!("Softmax.activate should not be called per-neuron; use Layer for vector execution");
            }
        }
    }

    pub fn deriv(&self, z: f64) -> f64 {
        match self {
            Activation::Linear => 1.0,
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-z).exp());
                s * (1.0 - s)
            }
            Activation::ReLU => {
                if z < 0.0 { 0.0 } else { 1.0 }
            }
            Activation::Softmax => {
                panic!("Softmax.deriv should be handled with CEE");
            }
        }
    }
}


/*
 * Neuron 1つの構造体
 * weights: 重み
 * bias: バイアス
 */
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    /* 
     * w*x + b を返す
     */
    fn composition(&self, input: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(input)
            .map(|(w, x)| w*x)
            .sum::<f64>()
        + self.bias
    }

    pub fn new(input_size: usize, init_range: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64>
            = (0..input_size)
            .map(|_| rng.gen_range(-init_range..init_range))
            .collect();
        let bias: f64 = rng.gen_range(-init_range..init_range);

        Self {
            weights,
            bias,
        }
    }
}



/*
 * 層の構造体
 * neurons: 層に所属しているニューロン全体
 * activation: 活性化関数
 */
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        assert!(output_size > 0, "Layer must have at least one node");

        let mut neurons: Vec<Neuron> = Vec::with_capacity(output_size);
        let init_range = match activation {
            Activation::Sigmoid => {
                (6.0 / (input_size + output_size) as f64).sqrt() // Xavier初期化: sqrt(1.0 / fan_in)
            }
            Activation::ReLU => {
                (2.0 / input_size as f64).sqrt() // He初期化: sqrt(2.0 / fan_in)
            }
            Activation::Softmax => {
                (6.0 / (input_size + output_size) as f64).sqrt() // Xavier相当 (???)
            }
            Activation::Linear => {
                0.0f64
            }
        };

        for _ in 0..output_size {
            neurons.push(Neuron::new(input_size, init_range));
        }

        Layer {
            neurons,
            activation,
        }
    }

    /*
     * 順伝播
     * input: 前層の出力
     */
    pub fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let z: Vec<f64> = self.neurons
            .iter()
            .map(|neuron| neuron.composition(input))
            .collect::<Vec<f64>>();

        let a: Vec<f64> = match self.activation {
            Activation::Softmax => {
                let max_z: f64 = z
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = z
                    .iter()
                    .map(|&z| (z - max_z).exp())
                    .collect::<Vec<f64>>();
                let sum_exp: f64 = exps.iter().sum();

                exps.iter().map(|&e| e / sum_exp).collect::<Vec<f64>>()
            }
            Activation::Sigmoid | Activation::ReLU | Activation::Linear => {
                z.iter()
                 .map(|&z| self.activation.activate(z))
                 .collect::<Vec<f64>>()
            }
        };

        (z, a)
    }
}

pub enum Loss {
    MeanSquareError,
    CrossEntropyError,
}

impl Loss {
    pub fn calculate(&self, output: &[f64], target: &[f64]) -> f64 {
        assert_eq!(output.len(), target.len(), "output and target mush have same length");
        match self {
            Loss::MeanSquareError => {
                let sum: f64 = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&o, &t)| (o - t).powi(2))
                    .sum();

                sum / (output.len() as f64)
            }
            Loss::CrossEntropyError => {
                let sum: f64 = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&o, &t)| {
                        let o_safe = o.max(1e-12);
                        -(t * o_safe.ln())
                    })
                    .sum();

                sum / (output.len() as f64)
            }
        }
    }

    pub fn delta(&self, output: &[f64], target: &[f64], activation: &Activation) -> Vec<f64> {
        assert_eq!(output.len(), target.len());

        match self {
            Loss::MeanSquareError => {
                output.iter()
                      .zip(target.iter())
                      .map(|(&o, &t)| o-t)
                      .collect::<Vec<f64>>()
            }
            Loss::CrossEntropyError => {
                match activation {
                    Activation::Softmax => {
                        output.iter()
                              .zip(target.iter())
                              .map(|(&o, &t)| o-t)
                              .collect::<Vec<f64>>()
                    }
                    _ => {
                        panic!("CEE Loss should pair with Softmax activation");
                    }
                }
            }
        }
    }
}

pub struct Model {
    pub layers: Vec<Layer>,
    pub loss_function: Loss,
}

impl Model {
    pub fn new(layer_sizes: &[usize], loss_function: Loss) -> Self {
        assert!(layer_sizes.len() >= 2, "layer_size must have at least input and output");

        let mut layers: Vec<Layer> = Vec::new();
        for idx in 0..layer_sizes.len()-1 {
            let in_size = layer_sizes[idx];
            let out_size = layer_sizes[idx+1];
            let activation = if idx+1 == layer_sizes.len()-1 {
                match loss_function {
                    Loss::MeanSquareError => Activation::Sigmoid,
                    Loss::CrossEntropyError => Activation::Softmax,
                }
            } else {
                Activation::Sigmoid
            };
            layers.push(Layer::new(in_size, out_size, activation));
        }
        Model {
            layers,
            loss_function,
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut a: Vec<f64> = input.to_vec();
        for layer in &self.layers {
            let (_z, a_new) = layer.forward(&a);
            a = a_new;
        }

        a
    }

    pub fn train(&mut self, input: &[f64], target: &[f64], lr: f64) -> f64 {
        // 順伝播で各層のz, aを記憶
        let mut activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len() + 1);
        let mut zs: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len() + 1);

        activations.push(input.to_vec());

        let mut a: Vec<f64> = input.to_vec();
        for layer in &self.layers {
            let (z_vec, a_vec): (Vec<f64>, Vec<f64>) = layer.forward(&a);
            zs.push(z_vec);
            a = a_vec;
            activations.push(a.clone());
        }

        let output: &Vec<f64> = activations.last().unwrap();

        // 逆伝搬を計算
        let last_activation = &self.layers.last().unwrap().activation;
        let mut delta = self.loss_function.delta(output, target, last_activation);

        // 各層を逆順に重みとバイアスを更新
        for layer_idx in (0..self.layers.len()).rev() {
            let z: &Vec<f64> = &zs[layer_idx];
            let a_prev = &activations[layer_idx];
            let is_output = layer_idx == self.layers.len() - 1;

            // 勾配計算
            let grad: Vec<f64> = if is_output {
                delta.clone()
            } else {
                delta.iter()
                     .zip(z.iter())
                     .map(|(&d, &z)| d * self.layers[layer_idx].activation.deriv(z))
                     .collect::<Vec<f64>>()
            };

            // 重みとバイアスの更新
            for neuron_idx in 0..self.layers[layer_idx].neurons.len() {
                self.layers[layer_idx].neurons[neuron_idx].bias -= lr * grad[neuron_idx];

                for weight_idx in 0..self.layers[layer_idx].neurons[neuron_idx].weights.len() {
                    self.layers[layer_idx].neurons[neuron_idx].weights[weight_idx] -= lr * grad[neuron_idx] * a_prev[weight_idx];
                }
            }

            // 逆伝搬を前の層に伝播
            if layer_idx > 0 {
                let prev_size = self.layers[layer_idx-1].neurons.len();
                let mut new_delta = vec![0.0; prev_size];

                for weight_idx in 0..prev_size {
                    let mut sum: f64 = 0.0;
                    for neuron_idx in 0..grad.len() {
                        sum += grad[neuron_idx] * self.layers[layer_idx].neurons[neuron_idx].weights[weight_idx];
                    }
                    new_delta[weight_idx] = sum;
                }
                delta = new_delta;
            }
        }

        // 損失を計算して返す
        return self.loss_function.calculate(output, target);
    }
}
