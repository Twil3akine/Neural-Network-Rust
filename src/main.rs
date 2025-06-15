mod utils;
use utils::*;

fn main() {
    let ceil: usize = 9;

    // モデルの構成: 入力2次元 -> 隠れ8ユニット -> 出力クラス数(2*ceil-1)
    let class_count = 2 * ceil - 1;
    let mut model: Model = Model::new(&[2, 32, 16, class_count], Loss::CrossEntropyError);

    let mut train_data: Vec<(Vec<f64>, Vec<f64>, usize)> = Vec::new();
    for x in 1..=ceil {
        for y in 1..=ceil {
            // 正解ラベル sum: 0 ..= 2*ceil-2
            let sum: usize = x + y - 2;

            if sum == (ceil*2 - 2) / 2 && x != 1 { continue; }

            let x01 = (x as f64 - 1.0) / ((ceil - 1) as f64); // [0,1]
            let y01 = (y as f64 - 1.0) / ((ceil - 1) as f64); // [0,1]

            let inp: Vec<f64> = vec![x01, y01];

            // ターゲット one-hot ベクトル長 class_count
            let mut target: Vec<f64> = vec![0.0; class_count];
            target[sum] = 1.0;

            train_data.push((inp, target, sum));
        }
    }


    // 学習パラメータ
    let epochs: usize = 2500;
    let mut lr: f64 = 0.01;

    // 学習ループ
    for epoch in 0..100 * epochs {
        let loss: f64 = train_data
            .iter()
            .map(|(input, target, _sum)| model.train(input, target, lr))
            .sum::<f64>();

        if epoch % epochs == 0 {
            println!("Epoch {:02}: loss = {:.3}", epoch / epochs + 1, loss / train_data.len() as f64);
            if (epoch / epochs) % 10 == 0 {
                lr *= 0.90;
            }
        }
    }

    // 結果表示
    println!("Result:");
    let mut confusion: Vec<Vec<usize>> = vec![vec![0usize; class_count]; class_count];
    for x in 1..=ceil {
        for y in 1..=ceil {
            let sum: usize = x + y - 2;
            let x01 = (x as f64 - 1.0) / ((ceil - 1) as f64);
            let y01 = (y as f64 - 1.0) / ((ceil - 1) as f64);
            let inp = vec![x01, y01];

            let output = model.forward(&inp);
            let (pred, _) = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            confusion[sum][pred] += 1;
        }
    }

    // 混同行列表示
    print!("    ");
    for i in 0..class_count {
        print!("{:^4}", i);
    }
    println!();
    for i in 0..class_count {
        print!("{:^4}", i);
        for j in 0..class_count {
            if confusion[i][j] == 0 {
                print!("{:^4}", confusion[i][j]);
            } else {
                print!("\x1b[36m{:^4}\x1b[037m", confusion[i][j]);
            }
        }
        println!();
    }
}

