use ndarray::prelude::*;
use crate::test_utils::read_data;
use crate::meanshift_parallel::MeanShiftParallel;


#[test]
fn test_runs_meanshift() {
    let dataset = read_data("data/wine.csv");
    let mut meanshift = MeanShiftParallel::new(20);
    let (cluster_centers, mut received_label) = meanshift.fit_predict(dataset);

    let expects: Array2<f32> = arr2(&[
        [1.26765306e+01, 2.56051020e+00, 2.34071429e+00, 2.06602040e+01,
        9.50000000e+01, 2.01673469e+00, 1.53959184e+00, 4.02857143e-01,
        1.40938776e+00, 5.00591836e+00, 9.01734694e-01, 2.30122449e+00,
        5.62153061e+02],
       [1.34051111e+01, 2.32866667e+00, 2.40444444e+00, 1.84844444e+01,
        1.08555556e+02, 2.48600000e+00, 2.28955556e+00, 3.19777778e-01,
        1.69977778e+00, 5.19977778e+00, 9.63466667e-01, 2.88466667e+00,
        9.15177778e+02],
       [1.37433333e+01, 1.89538462e+00, 2.42179487e+00, 1.71743589e+01,
        1.04384615e+02, 2.80589744e+00, 2.94512821e+00, 2.84871795e-01,
        1.85948718e+00, 5.45333333e+00, 1.07256410e+00, 3.11128205e+00,
        1.14230769e+03]
    ]);

    assert_eq!(expects[[0, 0]], cluster_centers[[0, 0]]);
    assert_eq!(expects[[0, 1]], cluster_centers[[0, 1]]);
    assert_eq!(expects[[0, 2]], cluster_centers[[0, 2]]);

    let expected_label = 0;
    received_label.dedup();
    assert_eq!(expected_label, received_label[0])
}
