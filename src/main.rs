#![allow(unused_imports)]

mod utils;

use utils::*;
use image::{ImageReader};
use std::fs;
use std::time::Instant;
use rand::Rng;
use rand::random;
use rand::seq::{SliceRandom, IndexedRandom};

const WIDTH: usize = 15;
const HEIGHT: usize = 15;
const IMAGE_SIZE: usize = WIDTH * HEIGHT;

fn load_dataset(dir: &str) -> Vec<(Vec<f64>, usize)> {
    let mut data = Vec::new();
    let entries = fs::read_dir(dir).expect("Designed directory is not exist.");

    for entry in entries {
        let entry = entry.expect("Reading file is failed.");
        let path = entry.path();

        if !path.is_file() { continue; }

        let label = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.split('-').next())
            .and_then(|s| s.parse::<usize>().ok())
            .expect("File name is not included label.");
        
        let img = ImageReader::open(&path)
            .expect("Reading image is failed.")
            .decode()
            .expect("Decoding is failed.")
            .to_luma8();

        let data_vec = img
            .pixels()
            .map(|p| p[0] as f64 / 255.0)
            .collect::<Vec<f64>>();

        assert_eq!(data_vec.len(), IMAGE_SIZE);
        data.push((data_vec, label - 1));
    }

    data
}



fn main() {
    let all_data = load_dataset("../images");
    let kind: usize = 5;
    let size_by_kind: usize = 40;
    let test_data_ratio: f64 = 0.8;
    let mut rng = rand::rng();
    
    let mut data_by_class = vec![Vec::new(); kind];
    for (input, label) in all_data {
        data_by_class[label].push((input, label));
    }

    // 学習データと検証データに分割
    let mut train_data = Vec::new();
    let mut test_data = Vec::new();
    for class_data in data_by_class {
        let (train, test): (Vec<_>, Vec<_>) = class_data
            .choose_multiple(&mut rng, size_by_kind)
            .cloned()
            .enumerate()
            .partition(|(i, _)| *i < (size_by_kind as f64 * test_data_ratio) as usize);
        train_data.extend(train.into_iter().map(|(_, d)| d));
        test_data.extend(test.into_iter().map(|(_, d)| d));
    }

    // 学習
    let mut model = Model::new(&[IMAGE_SIZE, 32, kind], Loss::CrossEntropyError);

    let epoch_size = 100;
    let mut lr = 0.01;

    let start_time: Instant = Instant::now();

    for epoch in 1..=epoch_size*100 {
        let loss: f64 = train_data
            .iter()
            .map(|(input, label)| {
                let mut target = vec![0.0; kind];
                target[*label] = 1.0;
                model.train(input, &target, lr)
            })
            .sum::<f64>();

        if epoch % epoch_size == 0 {
            let current_epoch: usize = epoch / epoch_size;
            let elapsed: f64 = start_time.elapsed().as_secs_f64();
            let avg_time_per_epoch: f64 = elapsed / current_epoch as f64;
            let remain_epochs: usize = 100 - current_epoch;
            let eta: f64 = avg_time_per_epoch * remain_epochs as f64;

            println!(
                "Epoch {:02}: Loss = {:.7} | Elapsed: {:.2}s | ETA: {:.2}s",
                current_epoch,
                loss / train_data.len() as f64,
                elapsed,
                eta,
            );

            if (epoch / epoch_size) / 10 == 0 { lr *= 0.90; }
        }
    }

    // 評価
    println!("\nResult:");
    let mut confusion = vec![vec![0usize; kind]; kind];
    for (input, label) in &test_data {
        let output = model.forward(input);
        let (pred, _confidence) = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        confusion[*label][pred] += 1;
    }

    println!("\nConfusion Matrix");
    print!("    ");

    for j in 0..kind { print!("{:^4}", j+1); }
    println!();
    for (i, row) in confusion.iter().enumerate() {
        print!("{:^4}", i+1);
        for &v in row {
            if v > 0 {
                print!("\x1b[036m{:^4}\x1b[0m", v);
            } else {
                print!("{:^4}", v);
            }
        }
        println!();
    }
}
