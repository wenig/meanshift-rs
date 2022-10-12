use crate::MeanShift;
use crate::distance_measure::{DTW, Euclidean, Manhattan};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::str::FromStr;
use std::sync::Arc;

type LibDataType = f64;


#[pyfunction]
fn meanshift_algorithm<'py>(
    py: Python<'py>,
    data: Vec<PyReadonlyArray1<'py, LibDataType>>,
    n_threads: usize,
    bandwidth: Option<LibDataType>,
    distance_measure: String,
) -> PyResult<(&'py Vec<PyArray1<LibDataType>>, Vec<i32>)> {
    let data = Arc::new(
        data.into_iter()
            .map(|x| x.as_array().to_owned()).collect()
    );

    let (labels, cluster_centers) = match distance_measure {
        Euclidean::name() => MeanShift::new(Euclidean, n_threads, bandwidth).cluster(data),
        Manhattan::name() => MeanShift::new(Manhattan, n_threads, bandwidth).cluster(data),
        DTW::name() => MeanShift::new(DTW, n_threads, bandwidth).cluster(data)
    }?;

    Ok((
        cluster_centers.into_iter().map(|x| x.into_pyarray(py)).collect(),
        labels
    ))
}

#[pymodule]
fn meanshift_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(meanshift_algorithm, m)?)?;

    Ok(())
}
