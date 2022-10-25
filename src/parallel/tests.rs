use crate::distance_measure::euclidean::Euclidean;
use crate::distance_measure::DTW;
use crate::parallel::MeanShift;
use crate::test_utils::{close_l1, read_data};
use ndarray::{arr2, Array2};

// todo: compare Arc<Vec<ArrayView1<f64>>> vs Vec<ArcArray1<f64>>

#[test]
fn test_parallel_meanshift() {
    let expects: Array2<f64> = arr2(&[[0.5185592, 0.43546146, 0.5697923]]);

    let mut mean_shift = MeanShift::<f64, Euclidean>::default();

    let dataset = read_data("data/test.csv");
    let (labels, centers) = mean_shift.cluster(dataset.view()).unwrap();

    assert_eq!(100, labels.len());
    assert_eq!(0, labels.into_iter().sum());

    close_l1(expects[[0, 0]], centers[0][0], 0.01);
    close_l1(expects[[0, 1]], centers[0][1], 0.01);
    close_l1(expects[[0, 2]], centers[0][2], 0.01);
}

#[test]
fn test_parallel_meanshift_dtw_runs_without_errors() {
    let mut mean_shift = MeanShift::<f64, DTW>::default();

    let dataset = read_data("data/test.csv");
    let (_labels, _centers) = mean_shift.cluster(dataset.view()).unwrap();
}
