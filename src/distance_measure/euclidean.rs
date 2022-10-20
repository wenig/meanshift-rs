use std::ops::Div;

use crate::distance_measure::DistanceMeasure;
use crate::utils::LibData;
use anyhow::{Error, Result};
use kdtree::distance::squared_euclidean;
use ndarray::{Array2, ArrayView2};

#[derive(Copy, Clone, Default)]
pub struct Euclidean;

impl<A: LibData> DistanceMeasure<A> for Euclidean {
    const NAME: &'static str = "euclidean";

    fn distance_slice(point_a: &[A], point_b: &[A]) -> A {
        squared_euclidean(point_a, point_b).sqrt()
    }

    fn distance(series_a: ArrayView2<A>, series_b: ArrayView2<A>) -> A {
        series_a
            .iter()
            .zip(series_b.iter())
            .map(|(a, b)| (*a - *b).powi(2))
            .sum::<A>()
            .sqrt()
    }

    fn mean(points: Vec<ArrayView2<A>>) -> Result<Array2<A>> {
        let el_shape = points
            .get(0)
            .ok_or_else(|| Error::msg("Empty points list"))?
            .shape();
        let el_len = el_shape[0];
        let el_d = el_shape[1];
        let el_n = points.len();
        let sum_vec = Array2::zeros([el_len, el_d]);
        Ok(points
            .into_iter()
            .fold(sum_vec, |a, b| a + b)
            .div(A::from_usize(el_n).unwrap()))
    }
}

#[cfg(test)]
mod test {
    use crate::distance_measure::Euclidean;
    use crate::DistanceMeasure;
    use ndarray::{arr2, Axis};

    #[test]
    fn test_distance_is_same() {
        let a = arr2(&[[0.0, 1.0, 2.0]]);
        let b = arr2(&[[3.0, 4.0, 5.0]]);

        assert_eq!(
            Euclidean::distance_slice(
                a.index_axis(Axis(0), 0).as_slice().unwrap(),
                b.index_axis(Axis(0), 0).as_slice().unwrap()
            ),
            Euclidean::distance(a.t(), b.t())
        )
    }
}
