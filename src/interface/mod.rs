pub use crate::interface::parameters::Parameters;
use ndarray::Array2;
use anyhow::Result;

mod parameters;

pub trait MeanShiftInterface {
    fn init(parameters: Parameters) -> Self;
    fn fit(&mut self, data: Array2<f32>) -> Result<MeanShiftResult>;
}

pub type MeanShiftResult = (Array2<f32>, Vec<usize>);
