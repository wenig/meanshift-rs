mod utils;
mod helper_functions;

use ndarray::{concatenate, Axis, Array2, ArrayView2, Array1};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter::FromIterator;
use kdtree::KdTree;
use num_traits::Float;
use std::sync::Arc;
use std::time::{SystemTime};
pub(crate) use crate::meanshift_base::utils::{RefArray, DistanceMeasure, SliceComp};
use log::*;
pub(crate) use crate::meanshift_base::helper_functions::{closest_distance, mean_shift_single};


#[derive(Default)]
pub(crate) struct MeanShiftBase {
    pub dataset: Option<Array2<f32>>,
    pub bandwidth: f32,
    pub means: Vec<(Array1<f32>, usize, usize, usize)>,
    pub cluster_centers: Option<Array2<f32>>,
    pub tree: Option<Arc<KdTree<f32, usize, RefArray>>>,
    pub center_tree: Option<KdTree<f32, usize, RefArray>>,
    pub labels: Vec<usize>,
    pub distance_measure: DistanceMeasure,
    pub start_time: Option<SystemTime>
}

impl MeanShiftBase {
    pub(crate) fn start_timer(&mut self) {
        self.start_time = Some(SystemTime::now());
    }

    pub(crate) fn end_timer(&mut self) {
        debug!("duration {}", SystemTime::now().duration_since(self.start_time.unwrap()).unwrap().as_millis());
    }

    pub(crate) fn build_center_tree(&mut self) {
        self.center_tree = Some(KdTree::new(self.dataset.as_ref().unwrap().shape()[1]));
    }

    pub(crate) fn estimate_bandwidth(&mut self) {
        let quantile = 0.3_f32;

        match &self.dataset {
            Some(data) => {

                let n_neighbors = (data.shape()[0] as f32 * quantile).max(1.0) as usize;

                let mut tree = KdTree::new(data.shape()[1]);
                for (i, point) in data.axis_iter(Axis(0)).enumerate() {
                    tree.add(RefArray(point.to_shared()), i).unwrap();
                }

                let bandwidth: f32 = data.axis_iter(Axis(0)).map(|x| {
                    let nearest = tree.nearest(x.to_slice().unwrap(), n_neighbors, &(self.distance_measure.call())).unwrap();
                    let sum = nearest.into_iter().map(|(dist, _)| dist).fold(f32::min_value(), f32::max);
                    sum.clone()
                }).sum();

                self.tree = Some(Arc::new(tree));
                self.bandwidth = bandwidth / data.shape()[0] as f32;
            },
            _ => panic!("Data not yet set!")
        }
    }

    pub(crate) fn collect_means(&mut self) {
        self.means.sort_by(|(a, a_intensity, _, _), (b, b_intensity, _, _)|
            {
                let intensity_cmp = a_intensity.cmp(b_intensity);
                match &intensity_cmp {
                    Ordering::Equal => {
                        a.slice_cmp(b).reverse()
                    },
                    _ => intensity_cmp.reverse()
                }
            }
        );

        self.means.dedup_by_key(|(x, _, _, _)| x.clone());

        let mut unique: HashMap<usize, bool> = HashMap::from_iter(self.means.iter().map(|(_, _, _, i)| (*i, true)));

        for (mean, _, _, i) in self.means.iter(){
            if unique[i] {
                let neighbor_idxs = self.center_tree.as_ref().unwrap().within(
                    mean.as_slice().unwrap(),
                    self.bandwidth,
                    &(self.distance_measure.call())).unwrap();
                for (_, neighbor) in neighbor_idxs {
                    match unique.get_mut(neighbor) {
                        None => {}
                        Some(val) => { *val = false}
                    }
                }
                *unique.get_mut(i).unwrap() = true;
            }
        }

        let dim = self.means[0].0.len();

        let cluster_centers: Vec<ArrayView2<f32>> =  self.means.iter().filter_map(|(mean, _, _, identifier)| {
            if unique[identifier] {
                Some(mean.view().into_shape((1, dim)).unwrap())
            } else {
                None
            }
        }).collect();

        self.cluster_centers = Some(concatenate(Axis(0), cluster_centers.as_slice()).unwrap());
    }

    fn mean_shift_single(&mut self, seed: usize, bandwidth: f32) -> (Array1<f32>, usize, usize) {
        mean_shift_single(
            self.dataset.as_ref().unwrap().to_shared(),
            self.tree.as_ref().unwrap().clone(),
            seed,
            bandwidth,
            self.distance_measure.clone()
        )
    }

    fn closest_distance(&mut self, point_id: usize) -> usize {
        closest_distance(
            self.dataset.as_ref().unwrap().to_shared(),
            point_id,
            self.cluster_centers.as_ref().unwrap().to_shared(),
            self.distance_measure.clone()
        )
    }
}
