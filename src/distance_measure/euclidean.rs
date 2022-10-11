use kdtree::distance::squared_euclidean;
use crate::distance_measure::DistanceMeasure;
use crate::utils::LibData;

#[derive(Copy, Clone, Default)]
pub struct Euclidean;

impl<A: LibData> DistanceMeasure<A> for Euclidean {
    fn distance(point_a: &[A], point_b: &[A]) -> A {
        squared_euclidean(point_a, point_b)
    }

    fn mean(points: Vec<&[A]>) -> &[A] {
        todo!()
    }
}
