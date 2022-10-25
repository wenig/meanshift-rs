use anyhow::Result;
use ndarray::{
    concatenate, s, ArcArray1, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis, ScalarOperand,
};
use num_traits::{Float, FromPrimitive};
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::str::FromStr;

pub trait LibData:
    'static
    + Unpin
    + Clone
    + Send
    + Default
    + Sync
    + Debug
    + Float
    + FromPrimitive
    + Sum
    + FromStr
    + ScalarOperand
    + Display
{
    const NAN: Self;
    const INFINITY: Self;
}

impl LibData for f32 {
    const NAN: Self = Self::NAN;
    const INFINITY: Self = Self::INFINITY;
}

impl LibData for f64 {
    const NAN: Self = Self::NAN;
    const INFINITY: Self = Self::INFINITY;
}

#[derive(Clone)]
pub struct RefArray<A: LibData>(pub ArcArray1<A>);

impl<A: LibData> AsRef<[A]> for RefArray<A> {
    fn as_ref(&self) -> &[A] {
        let array = &self.0;
        array.as_slice().unwrap()
    }
}

pub trait SliceComp {
    fn slice_cmp(&self, b: &Self) -> Ordering;
}

impl<A: LibData> SliceComp for Array1<A> {
    fn slice_cmp(&self, other: &Self) -> Ordering {
        debug_assert!(self.len() == other.len());
        let a = self.as_slice().unwrap();
        let b = other.as_slice().unwrap();
        for i in 0..b.len() {
            let cmp = a[i].partial_cmp(&b[i]).unwrap();
            if cmp.ne(&Ordering::Equal) {
                return cmp;
            }
        }
        Ordering::Equal
    }
}

pub fn nanmean<A: LibData>(arr: ArrayView3<A>, axis: Axis) -> Result<Array2<A>> {
    let mask = arr.mapv(|x| A::from_usize(!x.is_nan() as usize).unwrap());
    let nan_sum: Array2<A> =
        arr.axis_iter(axis)
            .fold(Array2::zeros([arr.shape()[1], arr.shape()[2]]), |res, s| {
                res + s.mapv(|x| {
                    if x.is_nan() {
                        A::from_usize(0).unwrap()
                    } else {
                        x
                    }
                })
            });
    let divisor = mask.sum_axis(axis);
    Ok(nan_sum / divisor)
}

/// Can only work with univariate time series for now.
pub fn time_series_to_matrix<A: LibData>(series: &Vec<ArrayView2<A>>) -> Array3<A> {
    let n_rows = series.len();
    let max_cols = series.iter().map(|s| s.len()).max().unwrap();
    let variates = series[0].shape()[1];
    let mut matrix: Array3<A> = Array3::zeros([n_rows, max_cols, variates]) + A::NAN;

    for (i, s) in series.iter().enumerate() {
        let rows = s.shape()[0];
        let mut row = matrix.slice_mut(s![i, ..rows, ..]);
        row.assign(s)
    }

    matrix
}

pub fn to_time_series_real_size<A: LibData>(series: ArrayView2<A>) -> Result<Array2<A>> {
    let array_views: Vec<ArrayView2<A>> = series
        .axis_iter(Axis(0))
        .filter_map(|p| {
            p.iter()
                .all(|x| !x.is_nan())
                .then(|| Some(p.insert_axis(Axis(0))))
        })
        .map(|x| x.unwrap())
        .collect();
    Ok(concatenate(Axis(0), &array_views)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};

    #[test]
    fn test_time_series_to_matrix() {
        let timeseries: Vec<Array2<f64>> = vec![arr2(&[[0.0, 1.0, 2.0]]), arr2(&[[3.0, 4.0]])];

        let matrix = time_series_to_matrix(&timeseries.iter().map(|x| x.t()).collect());

        println!("{:?}", matrix);

        assert_eq!(timeseries[0][[0, 0]], matrix[[0, 0, 0]]);
        assert_eq!(timeseries[0][[0, 1]], matrix[[0, 1, 0]]);
        assert!(matrix[[1, 2, 0]].is_nan());
    }

    #[test]
    fn test_to_time_series_real_size() {
        let timeseries: Vec<Array2<f64>> = vec![arr2(&[[0.0, 1.0, 2.0]]), arr2(&[[3.0, 4.0]])];

        let matrix = time_series_to_matrix(&timeseries.iter().map(|x| x.t()).collect());

        assert_eq!(
            to_time_series_real_size(matrix.index_axis(Axis(0), 0))
                .unwrap()
                .shape()[0],
            3
        );
        assert_eq!(
            to_time_series_real_size(matrix.index_axis(Axis(0), 1))
                .unwrap()
                .shape()[0],
            2
        );
    }

    #[test]
    fn test_nanmean() {
        let dataset = arr3(&[
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [4.0, 4.0], [f64::NAN, 5.0]],
        ]);

        let expected = arr2(&[[1.5, 1.5], [2.5, 2.5], [2.0, 3.5]]);

        let avg = nanmean(dataset.view(), Axis(0)).unwrap();
        assert_eq!(avg, expected)
    }
}
