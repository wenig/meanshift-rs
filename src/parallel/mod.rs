#[cfg(test)]
mod tests;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::sync::Arc;
use std::time::SystemTime;
use kdtree::KdTree;
use log::debug;
use ndarray::{ArcArray2, Array1, Array2, ArrayView1, ArrayView2, Axis, concatenate};
use rayon::prelude::*;
use crate::meanshift_base::{DistanceMeasure, LibData, RefArray};
use crate::meanshift_base::utils::SliceComp;


#[derive(Default)]
pub(crate) struct MeanShift<A: LibData> {
    pub bandwidth: Option<A>,
    pub cluster_centers: Option<Array2<A>>,
    pub tree: Option<Arc<KdTree<A, usize, RefArray<A>>>>,
    pub center_tree: Option<KdTree<A, usize, RefArray<A>>>,
    pub distance_measure: DistanceMeasure,
}

impl<A: LibData> MeanShift<A> {
    pub fn new(distance_measure: DistanceMeasure) -> Self {
        Self {
            bandwidth: None,
            cluster_centers: None,
            tree: None,
            center_tree: None,
            distance_measure,
        }
    }

    pub fn new_with_threads(distance_measure: DistanceMeasure, n_threads: usize) -> Self {
        //RAYON_NUM_THREADS todo: set env var with n_threads
        Self::new(distance_measure)
    }

    fn build_center_tree(&mut self, data: ArcArray2<A>) {
        let columns = data.shape()[1].clone();
        self.center_tree = Some(KdTree::new(columns));
    }

    fn estimate_bandwidth(&mut self, data: ArcArray2<A>) {
        match self.bandwidth {
            None => {
                let quantile = A::from(0.3).unwrap();
                let shape = (data.shape()).clone();
                let data_rows = A::from(shape[0]).unwrap();
                let one = A::from(1.0).unwrap();

                let n_neighbors: usize = A::from(data_rows * quantile)
                    .unwrap()
                    .max(one)
                    .to_usize()
                    .unwrap();

                let mut tree = KdTree::new(shape[1]);
                for (i, point) in data.axis_iter(Axis(0)).enumerate() {
                    tree.add(RefArray(point.to_shared()), i).unwrap();
                }

                let bandwidth: A = data
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|x| {
                        let nearest = tree
                            .nearest(
                                x.to_slice().unwrap(),
                                n_neighbors,
                                &self.distance_measure.optimized_call(),
                            )
                            .unwrap();
                        let sum = nearest
                            .into_iter()
                            .map(|(dist, _)| dist)
                            .fold(A::min_value(), A::max);
                        match &self.distance_measure {
                            DistanceMeasure::Minkowski => sum.sqrt(),
                            _ => sum,
                        }
                    })
                    .sum();

                self.tree = Some(Arc::new(tree));
                self.bandwidth = Some(bandwidth / data_rows);
            }
            _ => debug!("Skipping bandwidth estimation, because a bandwidth is already given."),
        }
    }

    fn collect_means(&mut self, mut means: Vec<(Array1<A>, usize, usize, usize)>) {
        means
            .sort_by(|(a, a_intensity, _, _), (b, b_intensity, _, _)| {
                let intensity_cmp = a_intensity.cmp(b_intensity);
                match &intensity_cmp {
                    Ordering::Equal => a.slice_cmp(b).reverse(),
                    _ => intensity_cmp.reverse(),
                }
            });

        means.dedup_by_key(|(x, _, _, _)| x.clone());

        let tree = self.center_tree.as_mut().unwrap();
        for (point, _, _, i) in means.iter() {
            tree.add(RefArray(point.to_shared()), *i).unwrap();
        }

        let mut unique: HashMap<usize, bool> =
            HashMap::from_iter(means.iter().map(|(_, _, _, i)| (*i, true)));

        let distance_fn = self.distance_measure.optimized_call();

        let two = A::from(2.0).unwrap();
        for (mean, _, _, i) in means.iter() {  // todo: parallelize
            if unique[i] {
                let neighbor_idxs = self.center_tree.as_ref().unwrap().within(
                    mean.as_slice().unwrap(),
                    self.bandwidth.expect("You must estimate or give a bandwidth before starting the algorithm!").powf(two),
                    &distance_fn).unwrap();
                for (_, neighbor) in neighbor_idxs {
                    match unique.get_mut(neighbor) {
                        None => {}
                        Some(val) => *val = false,
                    }
                }
                *unique.get_mut(i).unwrap() = true;
            }
        }

        let dim = means[0].0.len();

        let cluster_centers: Vec<ArrayView2<A>> = means
            .par_iter()
            .filter_map(|(mean, _, _, identifier)| {
                if unique[identifier] {
                    Some(mean.view().into_shape((1, dim)).unwrap())
                } else {
                    None
                }
            })
            .collect();

        self.cluster_centers = Some(concatenate(Axis(0), cluster_centers.as_slice()).unwrap());
    }

    fn label_data(&mut self, data: ArcArray2<A>) -> Vec<i32> {
        let cluster_centers = self.cluster_centers.as_ref().unwrap().to_shared();

        let labels: Vec<i32> = data.axis_iter(Axis(0)).into_par_iter()
            .map(|x| closest_distance(x, cluster_centers.clone(), self.distance_measure))
            .collect();
        labels
    }

    pub fn cluster(&mut self, dataset: ArcArray2<A>) -> Vec<i32> {
        self.estimate_bandwidth(dataset.clone());
        self.build_center_tree(dataset.clone());

        let shared_tree = self.tree.as_ref().unwrap();
        let bandwidth = self.bandwidth.as_ref().unwrap();

        let means: Vec<(Array1<A>, usize, usize)> = dataset.axis_iter(Axis(0)).into_par_iter().enumerate().map(|(i, point)|
            mean_shift_single(dataset.clone(), shared_tree.clone(), i, *bandwidth, self.distance_measure)
        )
            .filter(|(_, points_within_len, _)| points_within_len.gt(&0))
            .collect();

        let means: Vec<(Array1<A>, usize, usize, usize)> = means.into_iter().enumerate()
            .map(|(i, (means, points_within_len, iterations))| (means, points_within_len, iterations, i))
            .collect();

        self.collect_means(means);
        self.label_data(dataset)
    }
}


pub fn mean_shift_single<A: LibData>(
    data: ArcArray2<A>,
    tree: Arc<KdTree<A, usize, RefArray<A>>>,
    seed: usize,
    bandwidth: A,
    distance_measure: DistanceMeasure,
) -> (Array1<A>, usize, usize) {
    let stop_threshold = bandwidth.mul(A::from(1e-3).unwrap());
    let max_iter = 300;

    let mut my_mean = data.select(Axis(0), &[seed]).mean_axis(Axis(0)).unwrap();
    let mut iterations: usize = 0;
    let mut points_within_len: usize = 0;

    let distance_fn = distance_measure.optimized_call();
    let mean_fn = distance_measure.mean_call::<A>();
    let bandwidth = match &distance_measure {
        DistanceMeasure::Minkowski => bandwidth.powf(A::from(2.0).unwrap()),
        _ => bandwidth,
    };

    loop {
        let within_result = tree.within(my_mean.as_slice().unwrap(), bandwidth, &distance_fn);
        let neighbor_ids: Vec<usize> = match within_result {
            Ok(neighbors) => neighbors.into_iter().map(|(_, x)| *x).collect(),
            Err(_) => break,
        };

        let points_within = data.select(Axis(0), neighbor_ids.as_slice());
        points_within_len = points_within.shape()[0];
        let my_old_mean = my_mean;
        my_mean = points_within.mean_axis(Axis(0)).unwrap();

        if distance_measure.call()(my_mean.as_slice().unwrap(), my_old_mean.as_slice().unwrap())
            < stop_threshold
            || iterations >= max_iter
        {
            break;
        }

        iterations += 1;
    }

    (my_mean, points_within_len, iterations)
}

pub fn closest_distance<A: LibData>(
    data_point: ArrayView1<A>,
    cluster_centers: ArcArray2<A>,
    distance_measure: DistanceMeasure,
) -> i32 {
    let distance_fn = distance_measure.optimized_call();
    cluster_centers
        .axis_iter(Axis(0))
        .map(|center| distance_fn(data_point.as_slice().unwrap(), center.as_slice().unwrap()))
        .enumerate()
        .reduce(
            |(min_i, min), (i, x)| {
                if x < min {
                    (i, x)
                } else {
                    (min_i, min)
                }
            },
        )
        .unwrap()
        .0 as i32
}
