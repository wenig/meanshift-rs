mod meanshift_actors;
mod meanshift_base;
#[cfg(test)]
mod test_utils;
mod interface;

pub use meanshift_actors::MeanShiftActor;
pub use interface::{MeanShiftInterface, MeanShiftResult};

use pyo3::prelude::*;
use ndarray::{Array2};
use crate::interface::Parameters;
use crate::meanshift_base::{DistanceMeasure, LibDataType};
use std::str::FromStr;
use numpy::{PyArray2, IntoPyArray, PyReadonlyArray2};


#[pyfunction]
fn meanshift_algorithm<'py>(py: Python<'py>, data: PyReadonlyArray2<'py, LibDataType>, n_threads: usize, bandwidth: Option<LibDataType>, distance_measure: String) -> PyResult<(&'py PyArray2<LibDataType>, Vec<usize>)> {
    let data: Array2<LibDataType> = data.as_array().to_owned();
    let parameters = Parameters {
        n_threads,
        bandwidth,
        distance_measure: DistanceMeasure::from_str(&distance_measure)
            .expect(&format!("The distance measure '{}' does not exist! Use one of the following: 'minkowski', 'manhattan'",
                             &distance_measure))
    };

    let mut mean_shift = MeanShiftActor::init(parameters);
    let result = mean_shift.fit(data);

    let (array, labels) = result.expect("Could not fit a model to the given data!");
    Ok((array.into_pyarray(py), labels))
}


#[pymodule]
fn meanshift_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(meanshift_algorithm, m)?)?;

    Ok(())
}
