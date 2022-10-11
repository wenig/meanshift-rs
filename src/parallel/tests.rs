use std::sync::Arc;
use ndarray::{arr2, Array2, ArrayView1, Axis};
use crate::parallel::MeanShift;
use crate::test_utils::{close_l1, read_data};

// todo: compare Arc<Vec<ArrayView1<f64>>> vs Vec<ArcArray1<f64>>

#[test]
fn test_parallel_meanshift() {
    let mut mean_shift = MeanShift::<f64>::default();

    let dataset = read_data("data/test.csv");
    let array_vec: Vec<ArrayView1<f64>> = dataset.axis_iter(Axis(0)).collect();
    let labels = mean_shift.cluster(Arc::new(array_vec));

    assert_eq!(100, labels.len());
    assert_eq!(0, labels.into_iter().sum());

    let expects: Array2<f64> = arr2(&[[0.5185592, 0.43546146, 0.5697923]]);

    let centers = mean_shift.cluster_centers.unwrap();

    close_l1(expects[[0, 0]], centers[[0, 0]], 0.01);
    close_l1(expects[[0, 1]], centers[[0, 1]], 0.01);
    close_l1(expects[[0, 2]], centers[[0, 2]], 0.01);
}
