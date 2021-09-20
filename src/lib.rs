mod meanshift_actors;
mod meanshift_parallel;
mod meanshift_base;
#[cfg(test)]
mod test_utils;
#[cfg(test)]
mod tests;
mod interface;

pub use meanshift_actors::{MeanShiftActor, MeanShiftMessage, MeanShiftResponse};

use pyo3::prelude::*;
use ndarray::{Array2, arr2};


#[pyfunction]
fn meanshift_algorithm() -> PyResult<()> {

    Ok(())
}


#[pymodule]
fn meanshift_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(meanshift_algorithm, m)?)?;

    Ok(())
}
