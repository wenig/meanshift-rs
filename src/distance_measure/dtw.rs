use crate::distance_measure::DistanceMeasure;
use crate::utils::LibData;
use anyhow::Result;
use rand::seq::IteratorRandom;


#[derive(Copy, Clone, Default)]
pub struct DTW;

///from https://blog.acolyer.org/2016/05/13/dynamic-time-warping-averaging-of-time-series-allows-faster-and-more-accurate-classification/
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

    pub fn dba<A: LibData>(points: Vec<&[A]>, n_iterations: usize) -> Result<Vec<A>> {
        let mut center = points[Self::approximate_medoid(&points)];

        for i in 0..n_iterations {
            center = Self::dba_update();
        }

        Ok(center)
    }

    fn approximate_medoid<A: LibData>(points: &Vec<&[A]>) -> usize {
        let indices = if points.len() <= 50 {
            (0..points.len()).into_iter().collect()
        } else {
            let mut rng = &mut rand::thread_rng();
            (0..points.len()).choose_multiple(rng, 50)
        };

        indices.into_iter()
            .map(|i| (i, points.iter().map(|x| Self::dtw(points[i], *x).0).sum::<A>()))
            .min_by(|(_, sum_a), (_, sum_b)| sum_a.partial_cmp(sum_b).unwrap()).unwrap().0
    }

    fn dba_update<A: LibData>() -> Vec<A> {

    }
}

impl<A: LibData> DistanceMeasure<A> for DTW {
    fn distance(point_a: &[A], point_b: &[A]) -> A {
        DTW::dtw(point_a, point_b).0
    }

    fn mean(points: Vec<&[A]>) -> Result<Vec<A>> {
        DTW::dba(points, 10)
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
}
