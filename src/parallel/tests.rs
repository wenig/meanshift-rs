use ndarray::{arr2, Array2};
use crate::meanshift_base::LibData;
use crate::parallel::MeanShift;
use crate::test_utils::{close_l1, read_data};

#[test]
fn test_parallel_meanshift() {
    let mut mean_shift = MeanShift::<f64>::default();

    let dataset = read_data("data/test.csv");
    let labels = mean_shift.cluster(dataset.to_shared());

    assert_eq!(100, labels.len());
    assert_eq!(0, labels.into_iter().sum());

    let expects: Array2<f64> = arr2(&[[0.5185592, 0.43546146, 0.5697923]]);

    let centers = mean_shift.cluster_centers.unwrap();

    close_l1(expects[[0, 0]], centers[[0, 0]], 0.01);
    close_l1(expects[[0, 1]], centers[[0, 1]], 0.01);
    close_l1(expects[[0, 2]], centers[[0, 2]], 0.01);
}
