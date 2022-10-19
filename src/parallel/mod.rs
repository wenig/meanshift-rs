#[cfg(test)]
mod tests;

use anyhow::Result;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::sync::Arc;
use kdtree::KdTree;
use log::debug;
use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;
use crate::distance_measure::DistanceMeasure;
use crate::utils::{SliceComp, RefArray, LibData};


#[derive(Default)]
pub struct MeanShift<A: LibData, D: DistanceMeasure<A>> {
    pub bandwidth: Option<A>,
    pub cluster_centers: Option<Array2<A>>,
    pub tree: Option<Arc<KdTree<A, usize, RefArray<A>>>>,
    pub center_tree: Option<KdTree<A, usize, RefArray<A>>>,
    distance_measure: PhantomData<D>
}

impl<A: LibData, D: DistanceMeasure<A>> MeanShift<A, D> {
    pub fn new(_distance_measure: D, bandwidth: Option<A>) -> Self {
        Self {
            bandwidth,
            cluster_centers: None,
            tree: None,
            center_tree: None,
            distance_measure: PhantomData::default()
        }
    }

    pub fn new_with_threads(_distance_measure: D, bandwidth: Option<A>, n_threads: usize) -> Self {
        //RAYON_NUM_THREADS todo: set env var with n_threads
        Self::new(_distance_measure, bandwidth)
    }

    fn build_center_tree(&mut self, data: Arc<Vec<ArrayView1<A>>>) {
        let columns = data[0].len();
        self.center_tree = Some(KdTree::new(columns));
    }

    fn estimate_bandwidth(&mut self, data: Arc<Vec<ArrayView1<A>>>) {
        match self.bandwidth {
            None => {
                let quantile = A::from_f32(0.3).unwrap();
                let data_rows = A::from(data.len()).unwrap();
                let one = A::from_usize(1).unwrap();

                let n_neighbors: usize = A::from(data_rows * quantile)
                    .unwrap()
                    .max(one)
                    .to_usize()
                    .unwrap();

                let mut tree = KdTree::new(data[0].len());
                for (i, point) in data.iter().enumerate() {
                    tree.add(RefArray(point.to_shared()), i).unwrap();
                }

                let bandwidth: A = data
                    .par_iter()
                    .map(|x| {
                        let nearest = tree
                            .nearest(
                                x.to_slice().unwrap(),
                                n_neighbors,
                                &<D as DistanceMeasure<A>>::distance_slice
                            )
                            .unwrap();
                        nearest
                            .into_iter()
                            .map(|(dist, _)| dist)
                            .fold(A::min_value(), A::max)
                    })
                    .sum();

                self.tree = Some(Arc::new(tree));
                self.bandwidth = Some(bandwidth / data_rows);
            }
            _ => debug!("Skipping bandwidth estimation, because a bandwidth is already given."),
        }
    }

    fn collect_means(&mut self, mut means: Vec<(Array1<A>, usize, usize, usize)>) -> Vec<Array1<A>> {
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

        let distance_fn = &<D as DistanceMeasure<A>>::distance_slice;

        for (mean, _, _, i) in means.iter() {  // todo: parallelize
            if unique[i] {
                let neighbor_idxs = self.center_tree.as_ref().unwrap().within(
                    mean.as_slice().unwrap(),
                    self.bandwidth.expect("You must estimate or give a bandwidth before starting the algorithm!"),
                    distance_fn).unwrap();
                for (_, neighbor) in neighbor_idxs {
                    match unique.get_mut(neighbor) {
                        None => {}
                        Some(val) => *val = false,
                    }
                }
                *unique.get_mut(i).unwrap() = true;
            }
        }

        means
            .into_par_iter()
            .filter_map(|(mean, _, _, identifier)| {
                if unique[&identifier] {
                    Some(mean)
                } else {
                    None
                }
            })
            .collect()
    }

    fn label_data(&mut self, data: Arc<Vec<ArrayView1<A>>>, cluster_centers: Vec<ArrayView1<A>>) -> Vec<i32> {
        let labels: Vec<i32> = data.par_iter()
            .map(|x| closest_distance::<_, D>(x.clone(), cluster_centers.clone()))
            .collect();
        labels
    }

    pub fn cluster(&mut self, dataset: Arc<Vec<ArrayView1<A>>>) -> Result<(Vec<i32>, Vec<Array1<A>>)> {
        self.estimate_bandwidth(dataset.clone());
        self.build_center_tree(dataset.clone());

        let shared_tree = self.tree.as_ref().unwrap();
        let bandwidth = self.bandwidth.as_ref().unwrap();

        let means: Vec<(Array1<A>, usize, usize)> = dataset.par_iter().enumerate().map(|(i, _)|
            mean_shift_single::<_, D>(dataset.clone(), shared_tree.clone(), i, *bandwidth)
        )
            .filter(|(_, points_within_len, _)| {
                points_within_len.gt(&0)
            })
            .collect();

        let means: Vec<(Array1<A>, usize, usize, usize)> = means.into_iter().enumerate()
            .map(|(i, (means, points_within_len, iterations))| (means, points_within_len, iterations, i))
            .collect();


        let cluster_centers = self.collect_means(means);
        Ok((self.label_data(dataset, cluster_centers.iter().map(|x| x.view()).collect()), cluster_centers))
    }
}


pub fn mean_shift_single<A: LibData, D: DistanceMeasure<A>>(
    data: Arc<Vec<ArrayView1<A>>>,
    tree: Arc<KdTree<A, usize, RefArray<A>>>,
    seed: usize,
    bandwidth: A
) -> (Array1<A>, usize, usize) {
    let stop_threshold = bandwidth.mul(A::from_f32(1e-3).unwrap());
    let max_iter = 300;

    let mut my_mean = data[seed].to_owned();
    let mut iterations: usize = 0;
    let mut points_within_len: usize = 0;

    let distance_fn = &<D as DistanceMeasure<A>>::distance_slice;
    let mean_fn = &<D as DistanceMeasure<A>>::mean_1d;
    let zero = Array1::zeros([my_mean.shape()[0]]);

    loop {
        let within_result = tree.within(my_mean.as_slice().unwrap(), bandwidth, &distance_fn);
        let neighbor_ids: Vec<usize> = match within_result {
            Ok(neighbors) => neighbors.into_iter().map(|(_, x)| *x).collect(),
            Err(_) => break,
        };

        let points_within: Vec<ArrayView1<A>> = neighbor_ids.into_iter().map(|i| data[i]).collect();
        points_within_len = points_within.len();
        let my_old_mean = my_mean;
        my_mean = mean_fn(points_within).unwrap_or_else(|_| zero.clone());

        if points_within_len == 0
            || distance_fn(my_mean.as_slice().unwrap(), my_old_mean.as_slice().unwrap()) < stop_threshold
            || iterations >= max_iter
        {
            break;
        }

        iterations += 1;
    }

    (my_mean, points_within_len, iterations)
}

pub fn closest_distance<A: LibData, D: DistanceMeasure<A>>(
    data_point: ArrayView1<A>,
    cluster_centers: Vec<ArrayView1<A>>
) -> i32 {
    let distance_fn = &<D as DistanceMeasure<A>>::distance_slice;

    cluster_centers
        .into_iter()
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
