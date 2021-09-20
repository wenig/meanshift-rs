#[cfg(test)]
mod tests;
mod interface;

use crate::meanshift_base::{MeanShiftBase, mean_shift_single, RefArray, closest_distance};
use ndarray::{Array2, Array1, ArcArray2};
use ndarray::parallel::prelude::*;



#[derive(Default)]
pub struct MeanShiftParallel {
    meanshift: MeanShiftBase,
}

impl MeanShiftParallel {
    pub fn new(n_threads: usize) -> Self {
        rayon::ThreadPoolBuilder::new().num_threads(n_threads).build_global().unwrap();
        MeanShiftParallel::default()
    }

    pub fn fit(&mut self, data: Array2<f32>) {
        self.meanshift.dataset = Some(data);
        self.meanshift.build_center_tree();
        self.meanshift.estimate_bandwidth();

        let tree = self.meanshift.tree.as_ref().unwrap().clone();
        let bandwidth = self.meanshift.bandwidth.as_ref().expect("You must estimate or give a bandwidth before starting the algorithm!").clone();
        let distance_measure = self.meanshift.distance_measure.clone();
        self.meanshift.means = match &self.meanshift.dataset {
            Some(data) => {
                let dataset = data.to_shared();
                (0..data.shape()[0])
                    .into_par_iter()
                    .map(|i| {
                        let (means, points_within_len, iterations) = mean_shift_single(
                            dataset.clone(),
                            tree.clone(),
                            i,
                            bandwidth,
                            distance_measure.clone()
                        );

                        (means, points_within_len, iterations, i)
                    })
                    .filter(|(_, points_within_len, _, _)| {
                        points_within_len.gt(&0)
                    })
                    .collect::<Vec<(Array1<f32>, usize, usize, usize)>>()
            },
            None => {
                panic!("There should be data set by now!")
            }
        };

        for (mean, _, _, identifier) in self.meanshift.means.iter() {
            self.meanshift.center_tree.as_mut().unwrap().add(RefArray(mean.to_shared()), identifier.clone()).unwrap();
        }

        self.meanshift.collect_means();
    }

    pub fn predict(&mut self, data: Option<ArcArray2<f32>>) -> Vec<usize> {
        let data = match data {
            Some(data) => data,
            None => self.meanshift.dataset.as_ref().expect("You should provide a dataset before you predict its labels!").to_shared()
        };

        let cluster_centers = self.meanshift.cluster_centers.as_ref().expect("You must fit your model before predicting!").to_shared();
        let distance_measure = self.meanshift.distance_measure.clone();

        (0..data.shape()[0])
            .into_par_iter()
            .map(|i| {
                closest_distance(data.clone(), i, cluster_centers.clone(), distance_measure.clone())
            })
            .collect()
    }

    pub fn fit_predict(&mut self, data: Array2<f32>) -> (Array2<f32>, Vec<usize>) {
        self.fit(data);
        let labels = self.predict(None);

        (self.meanshift.cluster_centers.as_ref().unwrap().clone(), labels)
    }
}
