use crate::utils::LibData;
use anyhow::Result;

pub mod dtw;
pub mod euclidean;
pub mod manhattan;

pub use dtw::DTW;
pub use euclidean::Euclidean;
pub use manhattan::Manhattan;
use ndarray::{ArrayView2, Array2, Array1, ArrayView1, Axis};

pub trait DistanceMeasure<A: LibData>
where
    Self: Default + Copy + Clone + Sync,
{
    const NAME: &'static str;

    fn distance_slice(series_a: &[A], series_b: &[A]) -> A;
    fn distance(series_a: ArrayView2<A>, series_b: ArrayView2<A>) -> A;
    fn mean(points: Vec<ArrayView2<A>>) -> Result<Array2<A>>;

    fn mean_1d(points: Vec<ArrayView1<A>>) -> Result<Array1<A>> {
        let points: Vec<ArrayView2<A>> = points.into_iter().map(|x| x.insert_axis(Axis(0))).collect();
        Ok(Self::mean(points)?.index_axis_move(Axis(0), 0))
    }
}
