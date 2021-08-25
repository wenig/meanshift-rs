use ndarray::{Axis, ArcArray2, Array1, ArcArray1};
use crate::meanshift_base::utils::{DistanceMeasure, RefArray};
use std::sync::Arc;
use kdtree::KdTree;

pub fn mean_shift_single(data: ArcArray2<f32>, tree: Arc<KdTree<f32, usize, RefArray>>, seed: usize, bandwidth: f32, distance_measure: DistanceMeasure) -> (Array1<f32>, usize, usize) {
    let stop_threshold = 1e-3 * bandwidth;
    let max_iter = 300;

    let mut my_mean = data.select(Axis(0), &[seed]).mean_axis(Axis(0)).unwrap();
    let mut my_old_mean = my_mean.clone();
    let mut iterations: usize = 0;
    let mut points_within_len: usize = 0;
    loop {
        let within_result = tree.within(my_mean.as_slice().unwrap(), bandwidth, &(distance_measure.call()));
        let neighbor_ids: Vec<usize> = match within_result {
            Ok(neighbors) => neighbors.into_iter().map(|(_, x)| x.clone()).collect(),
            Err(_) => break
        };

        let points_within = data.select(Axis(0), neighbor_ids.as_slice());
        points_within_len = points_within.shape()[0];
        my_old_mean = my_mean;
        my_mean = points_within.mean_axis(Axis(0)).unwrap();

        if distance_measure.call()(my_mean.as_slice().unwrap(), my_old_mean.as_slice().unwrap()) < stop_threshold || iterations >= max_iter {
            break
        }

        iterations += 1;
    }

    //println!("took {} microseconds", SystemTime::now().duration_since(start).unwrap().as_micros());

    (my_mean, points_within_len, iterations)
}

pub fn closest_distance(data: ArcArray2<f32>, point_id: usize, cluster_centers: ArcArray2<f32>, distance_measure: DistanceMeasure) -> usize {
    let point = data.select(Axis(0), &[point_id]).mean_axis(Axis(0)).unwrap();
    cluster_centers.axis_iter(Axis(0)).map(|center| {
        distance_measure.call()(point.as_slice().unwrap(), center.as_slice().unwrap())
    }).enumerate().reduce(|(min_i, min), (i, x)| {
        if x < min { (i, x) } else { (min_i, min) }
    }).unwrap().0
}