use kdtree::distance::squared_euclidean;
use std::ops::Sub;
use ndarray::{ArcArray1, Array1};
use std::cmp::Ordering;

#[derive(Clone)]
pub enum DistanceMeasure {
    Minkowski,
    #[allow(dead_code)]
    Manhattan
}

impl DistanceMeasure {
    pub fn call(&self) -> fn(&[f32], &[f32]) -> f32 {
        match self {
            Self::Minkowski => |a, b| {squared_euclidean(a, b).sqrt()},
            Self::Manhattan => |a, b| {
                a.iter().zip(b.iter()).map(|(a_, b_)| {
                    a_.sub(b_).abs()
                }).sum()
            }
        }
    }
}

impl Default for DistanceMeasure {
    fn default() -> Self {
        DistanceMeasure::Minkowski
    }
}


pub struct RefArray(pub ArcArray1<f32>);

impl AsRef<[f32]> for RefArray {
    fn as_ref(&self) -> &[f32] {
        let arc_array = &self.0;
        arc_array.as_slice().unwrap()
    }
}

pub trait SliceComp {
    fn slice_cmp(&self, b: &Self) -> Ordering;
}

impl SliceComp for Array1<f32> {
    fn slice_cmp(&self, other: &Self) -> Ordering {
        debug_assert!(self.len() == other.len());
        let a = self.as_slice().unwrap();
        let b = other.as_slice().unwrap();
        for i in 0..b.len() {
            let cmp = a[i].partial_cmp(&b[i]).unwrap();
            if cmp.ne(&Ordering::Equal) {
                return cmp
            }
        }
        Ordering::Equal
    }
}
