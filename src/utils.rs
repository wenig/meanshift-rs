use ndarray::{ArcArray1, Array1, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::iter::Sum;
use std::str::FromStr;

pub trait LibData:
    'static + Unpin + Clone + Send + Default + Sync + Debug + Float + FromPrimitive + Sum + FromStr + ScalarOperand
{}

impl LibData for f32 {}
impl LibData for f64 {}


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
