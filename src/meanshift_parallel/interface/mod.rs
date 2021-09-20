#[cfg(test)]
mod tests;

use crate::interface::{MeanShiftInterface, Parameters, MeanShiftResult};
use crate::meanshift_parallel::MeanShiftParallel;
use ndarray::Array2;
use crate::meanshift_base::MeanShiftBase;
use anyhow::Result;

impl MeanShiftInterface for MeanShiftParallel {
    fn init(parameters: Parameters) -> Self {
        rayon::ThreadPoolBuilder::new().num_threads(parameters.n_threads).build_global().unwrap();
        MeanShiftParallel {
            meanshift: MeanShiftBase {
                bandwidth: parameters.bandwidth,
                distance_measure: parameters.distance_measure,
                ..Default::default()
            }
        }
    }

    fn fit(&mut self, data: Array2<f32>) -> Result<MeanShiftResult> {
        Ok(self.fit_predict(data))
    }
}
