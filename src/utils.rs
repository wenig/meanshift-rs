use ndarray::{ArcArray1, Array1, ScalarOperand, Array2, Array3, s, Axis};
use num_traits::{Float, FromPrimitive, Zero};
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::Add;
use std::str::FromStr;

pub trait LibData:
    'static + Unpin + Clone + Send + Default + Sync + Debug + Float + FromPrimitive + Sum + FromStr + ScalarOperand + Display
{
    const NAN: Self;
}

impl LibData for f32 {
    const NAN: Self = Self::NAN;
}
impl LibData for f64 {
    const NAN: Self = Self::NAN;
}


#[derive(Clone)]
pub struct RefArray<A: LibData>(pub ArcArray1<A>);

impl<A: LibData> AsRef<[A]> for RefArray<A> {
    fn as_ref(&self) -> &[A] {
        let arc_array = &self.0;
        arc_array.as_slice().unwrap()
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

/// Can only work with univariate time series for now.
pub fn time_series_to_matrix<A: LibData>(series: &Vec<&[A]>) -> Array3<A> {
    let n_rows = series.len();
    let max_cols = series.iter().map(|s| s.len()).max().unwrap();
    let variates = 1;
    let mut matrix: Array3<A> = Array3::zeros([n_rows, max_cols, variates]) + A::NAN;

    for (i, s) in series.iter().enumerate() {
        let ts: Array1<A> = Array1::from_vec(s.to_vec());
        let rows = ts.shape()[0];
        let mut row = matrix.slice_mut(s![i, ..rows, 0_usize]);
        row.assign(&ts)
    }

    matrix
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_to_matrix() {
        let timeseries: Vec<&[f64]> = vec![
            &[0.0, 1.0, 2.0],
            &[3.0, 4.0]
        ];

        let matrix = time_series_to_matrix(&timeseries);
        println!("{:?}", matrix);

        assert_eq!(timeseries[0][0], matrix[[0, 0, 0]]);
        assert_eq!(timeseries[0][1], matrix[[0, 1, 0]]);
        assert!(matrix[[1, 2, 0]].is_nan());
    }
}
