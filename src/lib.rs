mod meanshift_actors;
mod meanshift_parallel;
mod meanshift_base;
#[cfg(test)]
mod test_utils;
mod interface;

pub use meanshift_actors::MeanShiftActor;
pub use meanshift_parallel::MeanShiftParallel;
pub use interface::{MeanShiftInterface, MeanShiftResult};

use pyo3::prelude::*;
use ndarray::{Array2, arr2};
use crate::interface::Parameters;
use crate::meanshift_base::DistanceMeasure;
use std::str::FromStr;
use numpy::{PyArray2, IntoPyArray, PyReadonlyArray, PyReadonlyArray2};


#[pyfunction]
fn meanshift_algorithm<'py>(py: Python<'py>, data: PyReadonlyArray2<'py, f32>, n_threads: usize, bandwidth: Option<f32>, distance_measure: String, use_actors: bool) -> PyResult<(&'py PyArray2<f32>, Vec<usize>)> {
    let data: Array2<f32> = data.as_array().to_owned();
    let parameters = Parameters {
        n_threads,
        bandwidth,
        distance_measure: DistanceMeasure::from_str(&distance_measure)
            .expect(&format!("The distance measure '{}' does not exist! Use one of the following: 'squared_euclidean', 'minkowski', 'manhattan'",
                             &distance_measure))
    };

    let result = if use_actors {
        let mut mean_shift = MeanShiftActor::init(parameters);
        mean_shift.fit(data)
    } else {
        let mut mean_shift = MeanShiftParallel::init(parameters);
        MeanShiftInterface::fit(&mut mean_shift, data)
    };

    let (array, labels) = result.expect("Could not fit a model to the given data!");
    Ok((array.into_pyarray(py), labels))
}


#[pymodule]
fn meanshift_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(meanshift_algorithm, m)?)?;

    Ok(())
}
