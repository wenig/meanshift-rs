use kdtree::distance::squared_euclidean;
use std::ops::Sub;
use ndarray::{ArcArray1, Array1};
use std::cmp::Ordering;
use std::str::FromStr;

pub type LibDataType = f64;

#[derive(Clone)]
pub enum DistanceMeasure {
    Minkowski,
    #[allow(dead_code)]
    Manhattan
}

impl DistanceMeasure {
    pub fn call(&self) -> fn(&[LibDataType], &[LibDataType]) -> LibDataType {
        match self {
            Self::Minkowski => |a, b| {squared_euclidean(a, b).sqrt()},
            Self::Manhattan => |a, b| {
                a.iter().zip(b.iter()).map(|(a_, b_)| {
                    a_.sub(b_).abs()
                }).sum()
            }
        }
    }
}

impl Default for DistanceMeasure {
    fn default() -> Self {
        DistanceMeasure::Minkowski
    }
}

impl FromStr for DistanceMeasure {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq("minkowski") {
            Ok(Self::Minkowski)
        } else if s.eq("manhattan") {
            Ok(Self::Manhattan)
        } else {
            Err(())
        }
    }
}

#[derive(Clone)]
pub struct RefArray(pub ArcArray1<LibDataType>);

impl AsRef<[LibDataType]> for RefArray {
    fn as_ref(&self) -> &[LibDataType] {
        let arc_array = &self.0;
        arc_array.as_slice().unwrap()
    }
}

pub trait SliceComp {
    fn slice_cmp(&self, b: &Self) -> Ordering;
}

impl SliceComp for Array1<LibDataType> {
    fn slice_cmp(&self, other: &Self) -> Ordering {
        debug_assert!(self.len() == other.len());
        let a = self.as_slice().unwrap();
        let b = other.as_slice().unwrap();
        for i in 0..b.len() {
            let cmp = a[i].partial_cmp(&b[i]).unwrap();
            if cmp.ne(&Ordering::Equal) {
                return cmp
            }
        }
        Ordering::Equal
    }
}
