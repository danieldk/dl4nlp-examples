#![feature(test)]

extern crate boxed;
extern crate rand;
extern crate test;

use rand::Rng;
use rand::weak_rng;
use rand::distributions::{IndependentSample, Normal};

use test::{Bencher, black_box};

const ARRAY_SIZE: usize = 5000000;

fn unboxed_array(n: usize) -> Vec<f32> {
    let mut rng = weak_rng();
    let normal = Normal::new(0.0, 0.5);
    let mut arr = Vec::with_capacity(n);

    for _ in 0..n {
        arr.push(normal.ind_sample(&mut rng) as f32);
    }

    arr
}

fn boxed_array(n: usize) -> Vec<Box<f32>> {
    let mut rng = weak_rng();
    let normal = Normal::new(0.0, 0.5);
    let mut arr = Vec::with_capacity(n);

    for _ in 0..n {
        arr.push(Box::new(normal.ind_sample(&mut rng) as f32));
    }

    arr
}

#[bench]
fn simd_dp(b: &mut Bencher) {
    let arr = unboxed_array(ARRAY_SIZE);
    let arr2 = unboxed_array(ARRAY_SIZE);
    b.iter(|| black_box(boxed::dot(&arr, &arr2)))
}

#[bench]
fn simd_dp_shuffled(b: &mut Bencher) {
    let mut arr = unboxed_array(ARRAY_SIZE);
    let mut arr2 = unboxed_array(ARRAY_SIZE);

    let mut rng = rand::thread_rng();
    rng.shuffle(&mut arr);
    rng.shuffle(&mut arr2);

    b.iter(|| black_box(boxed::dot(&arr, &arr2)))
}

#[bench]
fn unboxed_dp(b: &mut Bencher) {
    let arr = unboxed_array(ARRAY_SIZE);
    let arr2 = unboxed_array(ARRAY_SIZE);
    b.iter(|| black_box(boxed::dot_unvectorized(&arr, &arr2)))
}

#[bench]
fn unboxed_dp_shuffled(b: &mut Bencher) {
    let mut arr = unboxed_array(ARRAY_SIZE);
    let mut arr2 = unboxed_array(ARRAY_SIZE);

    let mut rng = rand::thread_rng();
    rng.shuffle(&mut arr);
    rng.shuffle(&mut arr2);

    b.iter(|| black_box(boxed::dot_unvectorized(&arr, &arr2)))
}

#[bench]
fn boxed_dp(b: &mut Bencher) {
    let arr = boxed_array(ARRAY_SIZE);
    let arr2 = boxed_array(ARRAY_SIZE);
    b.iter(|| black_box(boxed::dot_unvectorized_boxed(&arr, &arr2)))
}

#[bench]
fn boxed_dp_shuffled(b: &mut Bencher) {
    let mut arr = boxed_array(ARRAY_SIZE);
    let mut arr2 = boxed_array(ARRAY_SIZE);

    let mut rng = rand::thread_rng();
    black_box(rng.shuffle(&mut arr));
    black_box(rng.shuffle(&mut arr2));

    b.iter(|| black_box(boxed::dot_unvectorized_boxed(&arr, &arr2)))
}
