pub use crate::interface::parameters::Parameters;
use crate::meanshift_base::LibDataType;
use ndarray::Array2;
use anyhow::Result;

mod parameters;

pub trait MeanShiftInterface {
    fn init(parameters: Parameters) -> Self;
    fn fit(&mut self, data: Array2<LibDataType>) -> Result<MeanShiftResult>;
}

pub type MeanShiftResult = (Array2<LibDataType>, Vec<usize>);
