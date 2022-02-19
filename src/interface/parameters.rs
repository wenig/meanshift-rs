use crate::meanshift_base::{DistanceMeasure};

pub struct Parameters<A> {
    pub n_threads: usize,
    pub bandwidth: Option<A>,
    pub distance_measure: DistanceMeasure
}

impl<A> Default for Parameters<A> {
    fn default() -> Self {
        Self {
            n_threads: 1,
            bandwidth: None,
            distance_measure: DistanceMeasure::default()
        }
    }
}
