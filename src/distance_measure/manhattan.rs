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

    fn distance(series_a: ArrayView2<A>, series_b: ArrayView2<A>) -> A {
        series_a.iter()
            .zip(series_b.iter())
            .map(|(a_, b_)| a_.sub(*b_).abs())
            .sum()
    }

    fn mean(points: Vec<ArrayView2<A>>) -> Result<Array2<A>> {
        Euclidean::mean(points)
    }

    fn name() -> String {
        "manhattan".to_string()
    }
}

#[cfg(test)]
mod test {
    use ndarray::{arr2, Axis};
    use crate::distance_measure::Manhattan;
    use crate::DistanceMeasure;

    #[test]
    fn test_distance_is_same() {
        let a = arr2(&[[0.0, 1.0, 2.0]]);
        let b = arr2(&[[3.0, 4.0, 5.0]]);

        assert_eq!(
            Manhattan::distance_slice(
                a.index_axis(Axis(0), 0).as_slice().unwrap(),
                b.index_axis(Axis(0), 0).as_slice().unwrap()
            ),
            Manhattan::distance(a.t(), b.t())
        )
    }
}
