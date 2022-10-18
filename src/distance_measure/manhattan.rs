use crate::distance_measure::{DistanceMeasure, Euclidean};
use crate::utils::LibData;
use anyhow::Result;
use ndarray::{Array2, ArrayView2};

#[derive(Copy, Clone, Default)]
pub struct Manhattan;

impl<A: LibData> DistanceMeasure<A> for Manhattan {
    fn distance_slice(point_a: &[A], point_b: &[A]) -> A {
        point_a.iter()
            .zip(point_b.iter())
            .map(|(a_, b_)| a_.sub(*b_).abs())
            .sum()
    }

    fn distance(series_a: ndarray::ArrayView2<A>, series_b: ndarray::ArrayView2<A>) -> A {
        todo!()
    }

    fn mean(points: Vec<ArrayView2<A>>) -> Result<Array2<A>> {
        Euclidean::mean(points)
    }

    fn name() -> String {
        "manhattan".to_string()
    }
}
