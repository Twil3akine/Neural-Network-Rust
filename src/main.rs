mod utils;

fn main() {
    let limit: usize = 100;
    let mut model = utils::Model::new(2, 16);

    let mut training_data: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for x in 0..15 {
        for y in 0..15 {
            let xor = x^y;
            let mut target = vec![0.0; 16];
            target[xor] = 1.0;
            training_data.push((vec![x as f64, y as f64], target));
        }
    }

    for epoch in 0..100*limit {
        let mut total_loss = 0.0;
        for (inputs, target) in &training_data {
            total_loss += model.train(inputs, target, 0.005);
        }

        if epoch % limit == 0 {
            let process: usize = epoch / limit + 1;
            println!(
                "Epoch {:02}: loss = {:.4}",
                process,
                total_loss / training_data.len() as f64
            );
        }
    }

    println!("\n");

    println!("Result:");

    let mut confusion_matrix: Vec<Vec<usize>> = vec![vec![0usize; 16]; 16];
    for (input, _target) in &training_data {
        let output = model.forward(input);
        let (predicted, _confidence) = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &v)| (i, v))
            .unwrap();

        let correct = (input[0] as usize) ^ (input[1] as usize);
        confusion_matrix[correct][predicted] += 1;
    }

    println!("Confusion Matrix:");
    print!("     ");

    for i in 0..16 {
        print!("{:^5}", i);
    }
    println!();

    for (i, row) in confusion_matrix.iter().enumerate() {
        print!("{:^5}", i);
        for val in row {
            print!("{:^5}", val);
        }
        println!();
    }
}
