use ndarray::{Array1, ArrayView2};
use crate::meanshift_base::LibData;

pub fn dba<A: LibData>(windows: ArrayView2<A>) -> Array1<A> {
    Array1::from_iter([A::from(1).unwrap()])
}
