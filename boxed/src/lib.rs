#![feature(cfg_target_feature)]

extern crate simd;

#[cfg(target_feature = "sse2")]
use simd::f32x4;

pub fn dot(v1: &[f32], v2: &[f32]) -> f32 {
    assert!(v1.len() == v2.len());

    let mut simd_sum = f32x4::splat(0f32);

    let mut i = 0;
    while i < v1.len() & !3 {
        let c1 = f32x4::load(v1, i);
        let c2 = f32x4::load(v2, i);
        simd_sum = simd_sum + c1 * c2;
        i += 4
    }

    let sum = simd_sum.extract(0) + simd_sum.extract(1) + simd_sum.extract(2) + simd_sum.extract(3);

    // Sum remaining components.
    let start = (v1.len() / 4) * 4;
    sum + dot_unvectorized(&v1[start..], &v2[start..])
}

#[cfg(not(target_feature = "sse2"))]
pub fn dot(v1: &[f32], v2: &[f32]) -> f32 {
    assert!(v1.len() == v2.len());

    dot_unvectorized(v1, v2)
}

pub fn dot_unvectorized(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).fold(0f32, |acc, (c1, c2)| acc + c1 * c2)
}

pub fn dot_unvectorized_boxed(v1: &[Box<f32>], v2: &[Box<f32>]) -> f32 {
    assert!(v1.len() == v2.len());

    v1.iter().zip(v2.iter()).fold(0f32, |acc, (v1, v2)| acc + **v1 * **v2)
}
