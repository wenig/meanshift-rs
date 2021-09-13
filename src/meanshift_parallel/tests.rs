use ndarray::prelude::*;
use log::*;
use crate::test_utils::read_data;
use crate::meanshift_parallel::MeanShiftParallel;


#[test]
fn test_runs_meanshift() {
    let dataset = read_data("data/test.csv");
    let mut meanshift = MeanShiftParallel::new(20);
    let (cluster_centers, mut received_label) = meanshift.fit_predict(dataset);

    let expects: Array2<f32> = arr2(&[
        [0.5185592, 0.43546146, 0.5697923]
    ]);

    assert_eq!(expects[[0, 0]], cluster_centers[[0, 0]]);
    assert_eq!(expects[[0, 1]], cluster_centers[[0, 1]]);
    assert_eq!(expects[[0, 2]], cluster_centers[[0, 2]]);

    let expected_label = 0;
    received_label.dedup();
    assert_eq!(expected_label, received_label[0])
}
