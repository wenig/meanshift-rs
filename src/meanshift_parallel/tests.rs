use ndarray::prelude::*;
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::str::FromStr;
use csv::{ReaderBuilder, Trim};
use log::*;
use crate::utils::read_data;
use crate::meanshift_parallel::MeanShiftParallel;


#[test]
fn test_runs_meanshift() {
    env_logger::init();

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
    debug!("received labels {:?}", received_label);
    assert_eq!(expected_label, received_label[0])
}
