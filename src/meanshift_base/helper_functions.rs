use ndarray::{Axis, ArcArray2, Array1};
use crate::meanshift_base::utils::{DistanceMeasure, RefArray};
use std::sync::Arc;
use kdtree::KdTree;
use crate::meanshift_base::LibData;

pub fn mean_shift_single<A: LibData>(data: ArcArray2<A>, tree: Arc<KdTree<A, usize, RefArray<A>>>, seed: usize, bandwidth: A, distance_measure: DistanceMeasure) -> (Array1<A>, usize, usize)

{
    let stop_threshold =  bandwidth.mul(A::from(1e-3).unwrap());
    let max_iter = 300;

    let mut my_mean = data.select(Axis(0), &[seed]).mean_axis(Axis(0)).unwrap();
    let mut iterations: usize = 0;
    let mut points_within_len: usize = 0;

    let distance_fn = distance_measure.optimized_call();
    let bandwidth = match &distance_measure {
        DistanceMeasure::Minkowski => bandwidth.powf(A::from(2.0).unwrap()),
        _ => bandwidth
    };

    loop {
        let within_result = tree.within(my_mean.as_slice().unwrap(), bandwidth, &distance_fn);
        let neighbor_ids: Vec<usize> = match within_result {
            Ok(neighbors) => neighbors.into_iter().map(|(_, x)| *x).collect(),
            Err(_) => break
        };

        let points_within = data.select(Axis(0), neighbor_ids.as_slice());
        points_within_len = points_within.shape()[0];
        let my_old_mean = my_mean;
        my_mean = points_within.mean_axis(Axis(0)).unwrap();

        if distance_measure.call()(my_mean.as_slice().unwrap(), my_old_mean.as_slice().unwrap()) < stop_threshold || iterations >= max_iter {
            break
        }

        iterations += 1;
    }

    //println!("took {} microseconds", SystemTime::now().duration_since(start).unwrap().as_micros());

    (my_mean, points_within_len, iterations)
}

pub fn closest_distance<A: LibData>(data: ArcArray2<A>, point_id: usize, cluster_centers: ArcArray2<A>, distance_measure: DistanceMeasure) -> usize {
    let distance_fn = distance_measure.optimized_call();
    let point = data.select(Axis(0), &[point_id]).mean_axis(Axis(0)).unwrap();
    cluster_centers.axis_iter(Axis(0)).map(|center| {
        distance_fn(point.as_slice().unwrap(), center.as_slice().unwrap())
    }).enumerate().reduce(|(min_i, min), (i, x)| {
        if x < min { (i, x) } else { (min_i, min) }
    }).unwrap().0
}
