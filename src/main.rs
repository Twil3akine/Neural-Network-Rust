mod utils;

fn main() {
    let mut neuron = utils::Neuron::new(2);

    let training_data: Vec<(Vec<f64>, f64)> = 
        vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 0.0),
            (vec![1.0, 0.0], 0.0),
            (vec![1.0, 1.0], 1.0),
        ];

    for epoch in 0..1_000_000_000 {
        for (inputs, target) in &training_data {
            neuron.train(inputs, *target, 0.005);
        }

        if epoch % (100_000) == 0 {
            println!("Epoch {epoch}");
        }
    }

    println!("テスト");
    for (inputs, _) in &training_data {
        let out = neuron.forward(inputs);
        println!("{:?} => {:.4}", inputs, out);
    }

    println!("Last weights: {:?}", neuron.weights);
    println!("Last bias: {:?}", neuron.bias);
}
