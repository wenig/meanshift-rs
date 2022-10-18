use std::ops::Div;

use kdtree::distance::squared_euclidean;
use ndarray::{ArrayView2, Array2};
use crate::distance_measure::DistanceMeasure;
use crate::utils::LibData;
use anyhow::{Result, Error};

#[derive(Copy, Clone, Default)]
pub struct Euclidean;

impl<A: LibData> DistanceMeasure<A> for Euclidean {
    fn distance_slice(point_a: &[A], point_b: &[A]) -> A {
        squared_euclidean(point_a, point_b).sqrt()
    }

    fn mean(points: Vec<ArrayView2<A>>) -> Result<Array2<A>> {
        let el_shape = points.get(0).ok_or_else(|| Error::msg("Empty points list"))?.shape();
        let el_len = el_shape[0];
        let el_d = el_shape[1];
        let el_n = points.len();
        let mut sum_vec = Array2::zeros([el_len, el_d]);
        Ok(points.into_iter().fold(sum_vec, |a, b| a + b).div(A::from_usize(el_n).unwrap()))
    }

    fn name() -> String {
        "euclidean".to_string()
    }

    fn distance(series_a: ndarray::ArrayView2<A>, series_b: ndarray::ArrayView2<A>) -> A {
        todo!()
    }
}
