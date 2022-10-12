use crate::utils::LibData;
use anyhow::Result;

pub mod euclidean;
pub mod manhattan;
pub mod dtw;

pub use euclidean::Euclidean;
pub use manhattan::Manhattan;
pub use dtw::DTW;

pub trait DistanceMeasure<A: LibData> where Self: Default + Copy + Clone + Sync {
    fn distance(point_a: &[A], point_b: &[A]) -> A;
    fn mean(points: Vec<&[A]>) -> Result<Vec<A>>;
    fn name() -> String;
}
