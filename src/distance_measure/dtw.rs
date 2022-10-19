use kdtree::distance::squared_euclidean;
use log::*;
use std::collections::HashMap;
use std::ops::Mul;
use crate::{distance_measure::DistanceMeasure, utils::time_series_to_matrix};
use crate::utils::{LibData, nanmean, to_time_series_real_size};
use anyhow::Result;
use ndarray::{Array2, Axis, Ix3, ArcArray, ArrayView2, s, Array1, ArrayView1, Array, arr2};

type ArcArray3<A> = ArcArray<A, Ix3>;

#[derive(Copy, Clone, Default)]
pub struct DTW;

///from https://github.com/tslearn-team/tslearn/blob/42a56cc/tslearn/barycenters/dba.py
impl DTW {
    fn cost_matrix<A: LibData>(point_a: ArrayView2<A>, point_b: ArrayView2<A>, mask: ArrayView2<A>) -> Array2<A> {
        let len_a = point_a.shape()[0];
        let len_b = point_b.shape()[0];
        let mut cum_sum = Array2::zeros([len_a + 1, len_b + 1]) + A::INFINITY;
        cum_sum[[0, 0]] = A::from_usize(0).unwrap();

        for i in 0..len_a {
            for j in 0..len_b {
                if mask[[i, j]].is_finite() {
                    cum_sum[[i + 1, j + 1]] = squared_euclidean(point_a.index_axis(Axis(0), i).as_slice().unwrap(), point_b.index_axis(Axis(0), j).as_slice().unwrap());
                    cum_sum[[i + 1, j + 1]] = cum_sum[[i + 1, j + 1]] + cum_sum[[i, j + 1]].min(cum_sum[[i + 1, j]]).min(cum_sum[[i, j]])
                }
            }
        }

        cum_sum.slice_move(s![1.., 1..])
    }

    fn return_path<A: LibData>(cost_matrix: ArrayView2<A>) -> Vec<(usize, usize)> {
        let mut path = vec![(cost_matrix.shape()[0] - 1, cost_matrix.shape()[1] - 1)];
        let (mut i, mut j) = (cost_matrix.shape()[0] - 1, cost_matrix.shape()[1] - 1);
        while (i, j) != (0, 0) {
            (i, j) = if i == 0 {
                (0, j - 1)
            } else if j == 0 {
                (i - 1, 0)
            } else {
                let arr = vec![
                    cost_matrix[[i - 1, j - 1]],
                    cost_matrix[[i - 1, j]],
                    cost_matrix[[i, j - 1]]
                ];
                match arr.into_iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 {
                    0 => (i - 1, j - 1),
                    1 => (i - 1, j),
                    _ => (i, j - 1)
                }
            };
            path.push((i, j))
        }
        path.reverse();
        path
    }

    /// from [tslearn](https://github.com/tslearn-team/tslearn/blob/42a56cc/tslearn/barycenters/dba.py)
    pub fn dba<A: LibData>(
            points: Vec<ArrayView2<A>>,
            barycenter_size: Option<usize>,
            init_barycenter: Option<Array2<A>>,
            max_iter: usize,
            tol: A,
            weights: Option<Array1<A>>,
            metric_params: Option<HashMap<String, A>>,
            n_init: usize
    ) -> Result<Array2<A>> {
        let mut best_cost = A::max_value();
        let mut best_center: Array2<A> = arr2(&[[]]);

        for _i in 0..n_init {
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
            points: &Vec<ArrayView2<A>>,
            mut barycenter_size: Option<usize>,
            init_barycenter: Option<Array2<A>>,
            max_iter: usize,
            tol: A,
            weights: Option<Array1<A>>,
            metric_params: Option<HashMap<String, A>>
    ) -> Result<(Array2<A>, A)> {
        let dataset = time_series_to_matrix(points);
        barycenter_size = barycenter_size.or_else(|| Some(dataset.shape()[1]));
        let weights = Self::set_weights(weights, dataset.shape()[0]);
        let mut barycenter = init_barycenter.unwrap_or_else(|| Self::init_avg(dataset.to_shared(), barycenter_size.unwrap()).unwrap());
        let mut cost_prev = A::max_value();
        let mut cost = A::max_value();
        for _i in 0..max_iter {
            let list_p_k;
            (list_p_k, cost) = Self::mm_assignment(dataset.to_shared(), barycenter.view(), weights.view(), metric_params.clone());
            let (diag_sum_v_k, list_w_k) = Self::mm_valence_warping(list_p_k, barycenter_size.unwrap(), weights.view());
            barycenter = Self::mm_update_barycenter(dataset.to_shared(), diag_sum_v_k, list_w_k)?;
            if (cost_prev - cost).abs() < tol {
                break
            } else if cost_prev < cost {
                warn!("DBA loss is increasing while it should not be. Stopping optimization.");
                break
            } else {
                cost_prev = cost
            }
        }

        Ok((barycenter, cost))
    }

    fn init_avg<A: LibData>(dataset: ArcArray3<A>, barycenter_size: usize) -> Result<Array2<A>> {
        if dataset.shape()[1] == barycenter_size {
            nanmean(dataset.view(), Axis(0))
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
        let _params = params.unwrap_or_else(|| HashMap::new());
        let n = dataset.shape()[0];
        let mut cost = A::from(0.0).unwrap();
        let mut list_p_k = vec![];
        for i in 0..n {
            let (path, dist_i) = Self::dtw_path(barycenter, to_time_series_real_size(dataset.index_axis(Axis(0), i)).unwrap().view());
            cost = cost + dist_i.powi(2).mul(weights[i]);
            list_p_k.push(path);
        }

        cost = cost.div(weights.sum());
        (list_p_k, cost)
    }

    fn mm_valence_warping<A: LibData>(list_p_k: Vec<Vec<(usize, usize)>>, barycenter_size: usize, weights: ArrayView1<A>) -> (Array1<A>, Vec<Array2<A>>) {
        let mut list_v_k = vec![];
        let mut list_w_k = vec![];
        let one = A::from_usize(1).unwrap();

        for (k, p_k) in list_p_k.into_iter().enumerate() {
            let sz_k = p_k[p_k.len() - 1].1 + 1;
            let mut w_k: Array2<A> = Array2::zeros([barycenter_size, sz_k]);
            for (i, j) in p_k {
                w_k[[i, j]] = one;
            }
            list_w_k.push(w_k.clone() * weights[k]);
            list_v_k.push(w_k.sum_axis(Axis(1)) * weights[k]);
        }

        let diag_sum_v_k = list_v_k.into_iter().fold(Array1::zeros([barycenter_size]), |a, b| a + b);

        (diag_sum_v_k, list_w_k)
    }

    fn mm_update_barycenter<A: LibData>(dataset: ArcArray3<A>, diag_sum_v_k: Array1<A>, list_w_k: Vec<Array2<A>>) -> Result<Array2<A>> {
        let d = dataset.shape()[2];
        let barycenter_size = diag_sum_v_k.shape()[0];
        let mut sum_w_x = Array2::zeros([barycenter_size, d]);
        for (_, (w_k, x_k)) in list_w_k.into_iter().zip(dataset.axis_iter(Axis(0))).enumerate() {
            sum_w_x = sum_w_x + w_k.dot(&to_time_series_real_size(x_k)?)
        }
        let into_diag = diag_sum_v_k.mapv(|x| x.powi(-1))
            .mul(Array2::ones([barycenter_size, 1]))
            .mul(Array::eye(barycenter_size));
        Ok(into_diag.dot(&sum_w_x))
    }

    pub fn dtw_path<A: LibData>(series_a: ArrayView2<A>, series_b: ArrayView2<A>) -> (Vec<(usize, usize)>, A) {
        let mask = Array2::zeros([series_a.shape()[0], series_b.shape()[0]]); // todo: use other masks such as sakoe_chiba, itakura
        let cost_matrix = Self::cost_matrix(series_a, series_b, mask.view());
        let path = Self::return_path(cost_matrix.view());
        (path, cost_matrix[[cost_matrix.shape()[0] - 1, cost_matrix.shape()[1] - 1]].sqrt())
    }
}

impl<A: LibData> DistanceMeasure<A> for DTW {
    const NAME: &'static str = "dtw";

    fn distance_slice(point_a: &[A], point_b: &[A]) -> A {
        DTW::dtw_path(
            ArrayView1::from(point_a).insert_axis(Axis(1)),
            ArrayView1::from(point_b).insert_axis(Axis(1))
        ).1
    }

    fn distance(series_a: ArrayView2<A>, series_b: ArrayView2<A>) -> A {
        DTW::dtw_path(series_a, series_b).1
    }

    fn mean(points: Vec<ArrayView2<A>>) -> Result<Array2<A>> {
        DTW::dba(points, None, None, 30, A::from_f32(0.00005).unwrap(), None, None, 1)
    }
}



#[cfg(test)]
mod tests {
    use ndarray::{Array2, arr2};

    use crate::distance_measure::dtw::DTW;
    use crate::DistanceMeasure;

    #[test]
    fn test_dtw_same_lengths() {
        let a = [0.94267613, 0.81582009, 0.63859374, 0.94131796, 0.67312447, 0.3352634 , 0.19988981, 0.3344863 , 0.77753481, 0.92335297];
        let b = [0.97218557, 0.56986568, 0.53248448, 0.67804195, 0.76575266, 0.19385823, 0.26328398, 0.44685084, 0.90686694, 0.75495287];

        let distance: f64 = DTW::distance_slice(&a, &b);
        assert!((distance - 1.28325005).abs() < 1e-7)
    }

    #[test]
    fn test_dtw_different_lengths() {
        let a = [0.16023953, 0.59981172, 0.86456616, 0.80691057, 0.1036448 , 0.48439886, 0.42657487, 0.17077501, 0.31801489, 0.90125957];
        let b = [0.37957006, 0.90812822, 0.2398513 , 0.23456596, 0.43191211];

        let distance: f64 = DTW::distance_slice(&a, &b);
        assert!((distance - 1.75368531).abs() < 1e-7)
    }

    #[test]
    fn test_dba_same_lengths() {
        let a = arr2(&[[0.94267613, 0.81582009, 0.63859374, 0.94131796, 0.67312447, 0.3352634 , 0.19988981, 0.3344863 , 0.77753481, 0.92335297]]);
        let b = arr2(&[[0.97218557, 0.56986568, 0.53248448, 0.67804195, 0.76575266, 0.19385823, 0.26328398, 0.44685084, 0.90686694, 0.75495287]]);

        let center = DTW::dba(vec![a.t(), b.t()], None, None, 30, 1e-5, None, None, 1).unwrap();

        let expected = arr2(&[[0.95743085],
                              [0.69284289],
                              [0.61637339],
                              [0.85353531],
                              [0.71943856],
                              [0.26456082],
                              [0.23158689],
                              [0.39066857],
                              [0.84220088],
                              [0.83915292_f64]]);

        for i in 0..a.len() {
            assert!((center[[i, 0]] - expected[[i, 0]]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_dba_different_lengths() {
        let a = arr2(&[[0.16023953, 0.59981172, 0.86456616, 0.80691057, 0.1036448 , 0.48439886, 0.42657487, 0.17077501, 0.31801489, 0.90125957]]);
        let b = arr2(&[[0.37957006, 0.90812822, 0.2398513 , 0.23456596, 0.43191211]]);

        let center = DTW::dba(vec![a.t(), b.t()], None, None, 30, 1e-5, None, None, 1).unwrap();

        let expected = arr2(&[[0.26990479],
                                               [0.79083537],
                                               [0.52338094],
                                               [0.52338094],
                                               [0.16910538],
                                               [0.45815549],
                                               [0.42924349],
                                               [0.30134356],
                                               [0.3749635 ],
                                               [0.66658584_f64]]);

        for i in 0..a.len() {
            assert!((center[[i, 0]] - expected[[i, 0]]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_cost_matrix() {
        let a = arr2(&[[0.94267613_f64, 0.81582009, 0.63859374, 0.94131796, 0.67312447, 0.3352634 , 0.19988981, 0.3344863 , 0.77753481, 0.92335297]]);
        let b = arr2(&[[0.97218557, 0.56986568, 0.53248448, 0.67804195, 0.76575266, 0.19385823, 0.26328398, 0.44685084, 0.90686694, 0.75495287]]);
        let mask = Array2::zeros([a.shape()[1], b.shape()[1]]);

        let matrix = DTW::cost_matrix(a.t(), b.t(), mask.view());

        assert!((matrix[[0, 0]] - 8.70807049e-04).abs() < 1e-9);
        assert!((matrix[[matrix.shape()[0]-1, matrix.shape()[1]-1]] - 1.63988435e-01).abs() < 1e-9)
    }

    #[test]
    fn test_return_path() {
        let a = arr2(&[[0.94267613_f64, 0.81582009, 0.63859374, 0.94131796, 0.67312447, 0.3352634 , 0.19988981, 0.3344863 , 0.77753481, 0.92335297]]);
        let b = arr2(&[[0.97218557, 0.56986568, 0.53248448, 0.67804195, 0.76575266, 0.19385823, 0.26328398, 0.44685084, 0.90686694, 0.75495287]]);
        let mask = Array2::zeros([a.shape()[1], b.shape()[1]]);

        let matrix = DTW::cost_matrix(a.t(), b.t(), mask.view());

        let path = DTW::return_path(matrix.view());

        assert_eq!(path, vec![(0, 0), (1, 0), (2, 1), (2, 2), (2, 3), (3, 4), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)])
    }

    #[test]
    fn test_dtw_path() {
        let a = arr2(&[[0.94267613_f64, 0.81582009, 0.63859374, 0.94131796, 0.67312447, 0.3352634 , 0.19988981, 0.3344863 , 0.77753481, 0.92335297]]);
        let b = arr2(&[[0.97218557, 0.56986568, 0.53248448, 0.67804195, 0.76575266, 0.19385823, 0.26328398, 0.44685084, 0.90686694, 0.75495287]]);

        let (path, distance) = DTW::dtw_path(a.t(), b.t());

        assert_eq!(path, [(0, 0), (1, 0), (2, 1), (2, 2), (2, 3), (3, 4), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]);
        assert_eq!(distance, 0.4049548559596511)
    }
}
