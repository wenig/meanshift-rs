use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::sync::Arc;
use std::time::SystemTime;
use futures::stream::iter;
use kdtree::KdTree;
use log::debug;
use ndarray::{ArcArray2, Array1, Array2, ArrayView2, Axis, concatenate};
use rayon::prelude::*;
use crate::meanshift_base::{DistanceMeasure, LibData, RefArray};


#[derive(Default)]
pub(crate) struct MeanShift<A: LibData> {
    pub dataset: Option<Array2<A>>,  // todo: must this be Option?
    pub bandwidth: Option<A>,
    pub means: Vec<(Array1<A>, usize, usize, usize)>,
    pub cluster_centers: Option<Array2<A>>,
    pub tree: Option<Arc<KdTree<A, usize, RefArray<A>>>>,
    pub center_tree: Option<KdTree<A, usize, RefArray<A>>>,
    pub distance_measure: DistanceMeasure,
    #[allow(dead_code)]
    pub start_time: Option<SystemTime>,
}

impl<A: LibData> MeanShift<A> {
    #[allow(dead_code)]
    pub(crate) fn start_timer(&mut self) {
        self.start_time = Some(SystemTime::now());
    }

    #[allow(dead_code)]
    pub(crate) fn end_timer(&mut self) {
        debug!(
            "duration {}",
            SystemTime::now()
                .duration_since(self.start_time.unwrap())
                .unwrap()
                .as_millis()
        );
    }

    pub(crate) fn build_center_tree(&mut self) {
        self.center_tree = Some(KdTree::new(self.dataset.as_ref().unwrap().shape()[1]));
    }

    pub(crate) fn estimate_bandwidth(&mut self) {
        match self.bandwidth {
            None => {
                let quantile = A::from(0.3).unwrap();

                match &self.dataset {
                    Some(data) => {
                        let data_rows = A::from(data.shape()[0]).unwrap();
                        let one = A::from(1.0).unwrap();

                        let n_neighbors: usize = A::from(data_rows * quantile)
                            .unwrap()
                            .max(one)
                            .to_usize()
                            .unwrap();

                        let mut tree = KdTree::new(data.shape()[1]);
                        for (i, point) in data.axis_iter(Axis(0)).enumerate() {
                            tree.add(RefArray(point.to_shared()), i).unwrap();
                        }

                        let bandwidth: A = data
                            .axis_iter(Axis(0))
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
                    _ => panic!("Data not yet set!"),
                }
            }
            _ => debug!("Skipping bandwidth estimation, because a bandwidth is already given."),
        }
    }

    pub(crate) fn collect_means(&mut self) {
        self.means
            .sort_by(|(a, a_intensity, _, _), (b, b_intensity, _, _)| {
                let intensity_cmp = a_intensity.cmp(b_intensity);
                match &intensity_cmp {
                    Ordering::Equal => a.slice_cmp(b).reverse(),
                    _ => intensity_cmp.reverse(),
                }
            });

        self.means.dedup_by_key(|(x, _, _, _)| x.clone());

        let tree = self.center_tree.as_mut().unwrap();
        for (point, _, _, i) in self.means.iter() {
            tree.add(RefArray(point.to_shared()), *i).unwrap();
        }

        let mut unique: HashMap<usize, bool> =
            HashMap::from_iter(self.means.iter().map(|(_, _, _, i)| (*i, true)));

        let distance_fn = self.distance_measure.optimized_call();

        let two = A::from(2.0).unwrap();
        for (mean, _, _, i) in self.means.iter() {  // todo: parallelize
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

        let dim = self.means[0].0.len();

        let cluster_centers: Vec<ArrayView2<A>> = self
            .means
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

    pub fn cluster(&mut self) -> () {
        self.estimate_bandwidth();
        self.build_center_tree();

        if let Some(dataset) = &self.dataset {
            let shared_dataset = dataset.to_shared();
            let shared_tree = self.tree.as_ref().unwrap();
            let bandwidth = self.bandwidth.as_ref().unwrap();

            self.means = dataset.axis_iter(Axis(0)).into_par_iter().enumerate().map(|(i, point)|
                mean_shift_single(shared_dataset.clone(), shared_tree.clone(), i, *bandwidth, self.distance_measure)
            )
                .filter(|(_, points_within_len, _)| points_within_len.gt(&0))
                .enumerate().map(|(i, (means, points_within_len, iterations))| (means, points_within_len, iterations, i))
                .collect();

            self.collect_means();

            // todo: labelling
        }
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
    let mean_fn = distance_measure.mean_call();
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

    //println!("took {} microseconds", SystemTime::now().duration_since(start).unwrap().as_micros());

    (my_mean, points_within_len, iterations)
}

pub fn closest_distance<A: LibData>(
    data: ArcArray2<A>,
    point_id: usize,
    cluster_centers: ArcArray2<A>,
    distance_measure: DistanceMeasure,
) -> usize {
    let distance_fn = distance_measure.optimized_call();
    let point = data
        .select(Axis(0), &[point_id])
        .mean_axis(Axis(0))
        .unwrap();
    cluster_centers
        .axis_iter(Axis(0))
        .map(|center| distance_fn(point.as_slice().unwrap(), center.as_slice().unwrap()))
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
        .0
}
