use kdtree::distance::squared_euclidean;
use crate::distance_measure::DistanceMeasure;
use crate::utils::LibData;
use anyhow::{Result, Error};

#[derive(Copy, Clone, Default)]
pub struct Euclidean;

impl<A: LibData> DistanceMeasure<A> for Euclidean {
    fn distance(point_a: &[A], point_b: &[A]) -> A {
        squared_euclidean(point_a, point_b).sqrt()
    }

    fn mean(points: Vec<&[A]>) -> Result<Vec<A>> {
        let el_len = points.get(0).ok_or_else(|| Error::msg("Empty points list"))?.len();
        let el_n = points.len();
        let mut sum_vec = vec![A::from(0).unwrap(); el_len];
        for point in points.into_iter() {
            for (i, p) in point.iter().enumerate() {
                let el = sum_vec.get_mut(i).unwrap();
                *el = el.add(*p);
            }
        }

        Ok(sum_vec.into_iter().map(|x| x.div(A::from(el_n).unwrap())).collect())
    }

    fn name() -> String {
        "euclidean".to_string()
    }
}
