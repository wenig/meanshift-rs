use crate::utils::LibData;

pub mod euclidean;
pub mod manhattan;
pub mod dtw;

pub trait DistanceMeasure<A: LibData> where Self: Default + Copy + Clone + Sync {
    fn distance(point_a: &[A], point_b: &[A]) -> A;
    fn mean(points: Vec<&[A]>) -> &[A];
}
