#![allow(dead_code)]

pub fn relu(x: isize) -> isize {
    if x < 0 { 0 } else { x }
}

pub fn step(x: isize) -> isize {
    if x < 0 { 0 } else { 1 }
}
