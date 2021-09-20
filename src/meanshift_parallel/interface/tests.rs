use crate::interface::{MeanShiftInterface, Parameters};
use crate::test_utils::read_data;
use ndarray::{Array2, arr2};
use crate::meanshift_parallel::MeanShiftParallel;

#[test]
fn test_interface_for_parallel() {
    let parameters = Parameters::default();
    let mut mean_shift = MeanShiftParallel::init(parameters);

    let dataset = read_data("data/test.csv");
    let (centers, labels) = MeanShiftInterface::fit(&mut mean_shift, dataset).expect("No MeanShiftResult was returned!");

    let expects: Array2<f32> = arr2(&[
        [0.5185592, 0.43546146, 0.5697923]
    ]);

    assert_eq!(expects[[0, 0]], centers[[0, 0]]);
    assert_eq!(expects[[0, 1]], centers[[0, 1]]);
    assert_eq!(expects[[0, 2]], centers[[0, 2]]);

    let expected_label = 0;
    assert_eq!(expected_label, labels[0])
}
