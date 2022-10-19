use crate::MeanShift;
use crate::DistanceMeasure;
use crate::distance_measure::{DTW, Euclidean, Manhattan};
use ndarray::{Array1, Array2, ArrayView1};
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
) -> PyResult<(Vec<&'py PyArray1<LibDataType>>, Vec<i32>)> {
    let data: Vec<Array1<LibDataType>> = data.into_iter()
            .map(|x| x.as_array().to_owned()).collect();

    let arc_data: Arc<Vec<ArrayView1<LibDataType>>> = Arc::new(data.iter().map(|x| x.view()).collect());

    let (labels, cluster_centers) = match distance_measure.as_str() {
        <Euclidean as DistanceMeasure<LibDataType>>::NAME => MeanShift::new(Euclidean, n_threads, bandwidth).cluster(arc_data),
        <Manhattan as DistanceMeasure<LibDataType>>::NAME => MeanShift::new(Manhattan, n_threads, bandwidth).cluster(arc_data),
        <DTW as DistanceMeasure<LibDataType>>::NAME => MeanShift::new(DTW, n_threads, bandwidth).cluster(arc_data),
        &_ => panic!("Distance measure {} not known.", distance_measure)
    }.expect("Clustering failed.");

    let cluster_centers: Vec<&PyArray1<LibDataType>> = cluster_centers.into_iter().map(|x| x.into_pyarray(py)).collect();

    Ok((
        cluster_centers,
        labels
    ))
}

#[pymodule]
fn meanshift_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(meanshift_algorithm, m)?)?;

    Ok(())
}
