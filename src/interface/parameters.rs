use crate::meanshift_base::{DistanceMeasure, LibDataType};

pub struct Parameters {
    pub n_threads: usize,
    pub bandwidth: Option<LibDataType>,
    pub distance_measure: DistanceMeasure
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            n_threads: 1,
            bandwidth: None,
            distance_measure: DistanceMeasure::default()
        }
    }
}
