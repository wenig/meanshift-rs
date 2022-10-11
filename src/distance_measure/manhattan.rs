use crate::distance_measure::DistanceMeasure;
use crate::utils::LibData;

#[derive(Copy, Clone, Default)]
pub struct Manhattan;

impl<A: LibData> DistanceMeasure<A> for Manhattan {
    fn distance(point_a: &[A], point_b: &[A]) -> A {
        point_a.iter()
            .zip(point_b.iter())
            .map(|(a_, b_)| a_.sub(*b_).abs())
            .sum()
    }

    fn mean(points: Vec<&[A]>) -> &[A] {
        todo!()
    }
}