use log::*;
use std::cmp::min_by;
use std::collections::HashMap;
use crate::{distance_measure::DistanceMeasure, utils::time_series_to_matrix};
use crate::utils::LibData;
use anyhow::{Error, Result};
use ndarray::{Array3, Array2, Axis, Ix3, ArcArray, ArrayView2, s, Array1, ArrayView1};
use rand::seq::IteratorRandom;

type Idx = (usize, usize);
type ArcArray3<A> = ArcArray<A, Ix3>;

#[derive(Copy, Clone, Default)]
pub struct DTW;

///from https://github.com/tslearn-team/tslearn/blob/42a56cc/tslearn/barycenters/dba.py
impl DTW {
    pub fn dtw<A: LibData>(point_a: &[A], point_b: &[A]) -> (A, Vec<Vec<A>>) {
        let mut cost_matrix = vec![vec![A::max_value(); point_b.len()+1]; point_a.len()+1];
        cost_matrix[0][0] = A::from(0.).unwrap();

        for i in 1..point_a.len()+1 {
            for j in 1..point_b.len()+1 {
                let cost = (point_a[i - 1] - point_b[j - 1]).abs();
                cost_matrix[i][j] = cost + A::min(
                    A::min(
                        cost_matrix[i - 1][j],
                        cost_matrix[i][j - 1]
                    ),
                    cost_matrix[i - 1][j - 1]
                );
            }
        }

        (cost_matrix[point_a.len()][point_b.len()], cost_matrix)
    }

    /// from [tslearn](https://github.com/tslearn-team/tslearn/blob/42a56cc/tslearn/barycenters/dba.py)
    pub fn dba<A: LibData>(
            points: Vec<&[A]>,
            barycenter_size: Option<usize>,
            init_barycenter: Option<Array2<A>>,
            max_iter: usize,
            tol: A,
            weights: Option<Array1<A>>,
            metric_params: Option<HashMap<String, A>>,
            n_init: usize
    ) -> Result<Vec<A>> {
        let mut best_cost = A::max_value();
        let mut best_center: Vec<A> = vec![];

        for i in 0..n_init {
            println!("Attempt {}", i+1);
            let (center, cost) = Self::dba_one_init(
                &points,
                barycenter_size,
                init_barycenter.clone(),
                max_iter,
                tol,
                weights.clone(),
                metric_params.clone()
            )?;
            if cost < best_cost {
                best_cost = cost;
                best_center = center;
            }
        }

        Ok(best_center)
    }

    /// todo: rm options for barycenter size etc
    pub fn dba_one_init<A: LibData>(
            points: &Vec<&[A]>,
            barycenter_size: Option<usize>,
            init_barycenter: Option<Array2<A>>,
            max_iter: usize,
            tol: A,
            weights: Option<Array1<A>>,
            metric_params: Option<HashMap<String, A>>
    ) -> Result<(Vec<A>, A)> {
        let dataset = time_series_to_matrix(points);
        barycenter_size = barycenter_size.or_else(|| Some(dataset.shape()[1]));
        let weights = Self::set_weights(weights, dataset.shape()[0]);
        let mut barycenter = init_barycenter.unwrap_or_else(|| Self::init_avg(dataset.to_shared(), barycenter_size.unwrap()).unwrap());
        let (mut cost_prev, cost) = (A::max_value(), A::max_value());
        for i in 0..max_iter {
            let (list_p_k, cost) = Self::mm_assignment(dataset.to_shared(), barycenter.view(), weights.view(), metric_params.clone());
            let (diag_sum_v_k, list_w_k) = Self::mm_valence_warping(list_p_k, barycenter_size.unwrap(), weights.view());
            println!("[DBA] epoch {i}, cost: {cost}");
            barycenter = Self::mm_update_barycenter(dataset.to_shared(), diag_sum_v_k, list_w_k);
            if (cost_prev - cost).abs() < tol {
                break
            } else if cost_prev < cost {
                warn!("DBA loss is increasing while it should not be. Stopping optimization.");
                break
            } else {
                cost_prev = cost
            }
        }

        Ok((vec![], A::max_value()))
    }

    fn init_avg<A: LibData>(dataset: ArcArray3<A>, barycenter_size: usize) -> Result<Array2<A>> {
        if dataset.shape()[1] == barycenter_size {
            dataset.mean_axis(Axis(0)).ok_or_else(|| Error::msg("Dataset is empty"))
        } else {
            todo!()
        }
    }

    fn set_weights<A: LibData>(weights: Option<Array1<A>>, n: usize) -> Array1<A> {
        match weights {
            Some(w) => w,
            None => Array1::ones([n]),
        }
    }

    fn mm_assignment<A: LibData>(dataset: ArcArray3<A>, barycenter: ArrayView2<A>, weights: ArrayView1<A>, params: Option<HashMap<String, A>>) -> (Vec<Vec<(usize, usize)>>, A) {
        let params = params.unwrap_or_else(|| HashMap::new());
        let n = dataset.shape()[0];
        let mut cost = A::from(0.0).unwrap();
        let mut list_p_k = vec![];
        for i in 0..n {
            let (path, dist_i) = Self::dtw_path(barycenter, dataset.index_axis(Axis(0), i), params);
            cost = cost + dist_i.powi(2).mul(weights[i]);
            list_p_k.push(path);
        }

        cost = cost.div(weights.sum());
        (list_p_k, cost)
    }

    fn mm_valence_warping(list_p_k: ) {
        todo!()
    }

    fn mm_update_barycenter() {
        todo!()
    }

    fn dtw_path<A: LibData>(barycenter: ArrayView2<A>, series: ArrayView2<A>, metric_params: HashMap<String, A>) -> (Vec<(usize, usize)>, A) {
        todo!()
    }
}

impl<A: LibData> DistanceMeasure<A> for DTW {
    fn distance(point_a: &[A], point_b: &[A]) -> A {
        DTW::dtw(point_a, point_b).0
    }

    fn mean(points: Vec<&[A]>) -> Result<Vec<A>> {
        DTW::dba(points, None, None, 30, A::from(0.00005).unwrap(), None, None, 1)
    }

    fn name() -> String {
        "dtw".to_string()
    }
}



#[cfg(test)]
mod tests {
    use crate::distance_measure::dtw::DTW;
    use crate::DistanceMeasure;

    #[test]
    fn test_dtw_same_lengths() {
        let a = [0.94267613, 0.81582009, 0.63859374, 0.94131796, 0.67312447, 0.3352634 , 0.19988981, 0.3344863 , 0.77753481, 0.92335297];
        let b = [0.97218557, 0.56986568, 0.53248448, 0.67804195, 0.76575266, 0.19385823, 0.26328398, 0.44685084, 0.90686694, 0.75495287];

        let distance: f64 = DTW::distance(&a, &b);
        assert!((distance - 1.28325005).abs() < 1e-7)
    }

    #[test]
    fn test_dtw_different_lengths() {
        let a = [0.16023953, 0.59981172, 0.86456616, 0.80691057, 0.1036448 , 0.48439886, 0.42657487, 0.17077501, 0.31801489, 0.90125957];
        let b = [0.37957006, 0.90812822, 0.2398513 , 0.23456596, 0.43191211];

        let distance: f64 = DTW::distance(&a, &b);
        assert!((distance - 1.75368531).abs() < 1e-7)
    }

    #[test]
    fn test_dba_same_lengths() {
        let a: [f64; 10] = [0.94267613, 0.81582009, 0.63859374, 0.94131796, 0.67312447, 0.3352634 , 0.19988981, 0.3344863 , 0.77753481, 0.92335297];
        let b = [0.97218557, 0.56986568, 0.53248448, 0.67804195, 0.76575266, 0.19385823, 0.26328398, 0.44685084, 0.90686694, 0.75495287];

        //let center = DTW::dba(vec![&a, &b], 10);
        //println!("{:?}", center);
    }
}
