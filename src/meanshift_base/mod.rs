mod helper_functions;
mod utils;

pub(crate) use crate::meanshift_base::helper_functions::{closest_distance, mean_shift_single};
pub(crate) use crate::meanshift_base::utils::LibData;
pub(crate) use crate::meanshift_base::utils::{DistanceMeasure, RefArray, SliceComp};
use kdtree::KdTree;
use log::*;
use ndarray::{concatenate, Array1, Array2, ArrayView2, Axis};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::sync::Arc;
use std::time::SystemTime;

#[derive(Default)]
pub(crate) struct MeanShiftBase<A: LibData> {
    pub dataset: Option<Array2<A>>,
    pub bandwidth: Option<A>,
    pub means: Vec<(Array1<A>, usize, usize, usize)>,
    pub cluster_centers: Option<Array2<A>>,
    pub tree: Option<Arc<KdTree<A, usize, RefArray<A>>>>,
    pub center_tree: Option<KdTree<A, usize, RefArray<A>>>,
    pub distance_measure: DistanceMeasure,
    #[allow(dead_code)]
    pub start_time: Option<SystemTime>,
}

impl<A: LibData> MeanShiftBase<A> {
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
        for (mean, _, _, i) in self.means.iter() {
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
            .iter()
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
}

impl<A: LibData> Clone for MeanShiftBase<A> {
    fn clone(&self) -> Self {
        Self {
            dataset: self.dataset.clone(),
            bandwidth: self.bandwidth,
            means: self.means.clone(),
            cluster_centers: self.cluster_centers.clone(),
            tree: None,
            center_tree: None,
            distance_measure: self.distance_measure.clone(),
            start_time: self.start_time,
        }
    }
}
