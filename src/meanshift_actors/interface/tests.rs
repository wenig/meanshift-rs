use crate::MeanShiftActor;
use crate::interface::{MeanShiftInterface, Parameters};
use crate::test_utils::read_data;
use ndarray::{Array2, arr2};

#[test]
fn test_interface_for_actor() {
    let parameters = Parameters::default();
    let mut mean_shift = MeanShiftActor::init(parameters);

    let dataset = read_data("data/test.csv");
    let (centers, labels) = mean_shift.fit(dataset).expect("No MeanShiftResult was returned!");

    let expects: Array2<f32> = arr2(&[
        [0.5185592, 0.43546146, 0.5697923]
    ]);

    assert_eq!(expects[[0, 0]], centers[[0, 0]]);
    assert_eq!(expects[[0, 1]], centers[[0, 1]]);
    assert_eq!(expects[[0, 2]], centers[[0, 2]]);

    let expected_label = 0;
    assert_eq!(expected_label, labels[0])
}
