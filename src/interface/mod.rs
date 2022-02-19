pub use crate::interface::parameters::Parameters;
use ndarray::Array2;
use anyhow::Result;

mod parameters;

pub trait MeanShiftInterface<A> {
    fn init(parameters: Parameters<A>) -> Self;
    fn fit(&mut self, data: Array2<A>) -> Result<MeanShiftResult<A>>;
}

pub type MeanShiftResult<A> = (Array2<A>, Vec<usize>);
