use crate::meanshift_base::DistanceMeasure;

pub struct Parameters {
    pub n_threads: usize,
    pub bandwidth: Option<f32>,
    pub distance_measure: DistanceMeasure
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            n_threads: 1,
            bandwidth: None,
            distance_measure: DistanceMeasure::SquaredEuclidean
        }
    }
}
