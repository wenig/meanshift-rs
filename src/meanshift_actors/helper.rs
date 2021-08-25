use actix::{Actor, SyncContext, Handler, ActorContext};
use crate::meanshift_actors::messages::{MeanShiftHelperWorkMessage, PoisonPill};

use ndarray::prelude::*;
use kdtree::{KdTree};
use ndarray::{ArcArray2};

use crate::meanshift_actors::{MeanShiftHelperResponse};
use std::sync::Arc;
use crate::meanshift_base::{mean_shift_single, RefArray, DistanceMeasure};


pub struct MeanShiftHelper {
    data: ArcArray2<f32>,
    tree: Arc<KdTree<f32, usize, RefArray>>,
    bandwidth: f32,
    distance_measure: DistanceMeasure
}

impl MeanShiftHelper {
    pub fn new(data: ArcArray2<f32>, tree: Arc<KdTree<f32, usize, RefArray>>, bandwidth: f32, distance_measure: DistanceMeasure) -> Self {
        Self {
            data,
            tree,
            bandwidth,
            distance_measure
        }
    }

    fn mean_shift_single(&mut self, seed: usize, bandwidth: f32) -> (Array1<f32>, usize, usize) {
        mean_shift_single(
            self.data.to_shared(),
            self.tree.clone(),
            seed,
            bandwidth,
            self.distance_measure.clone()
        )
    }
}

impl Actor for MeanShiftHelper {
    type Context = SyncContext<Self>;
}

impl Handler<MeanShiftHelperWorkMessage> for MeanShiftHelper {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftHelperWorkMessage, ctx: &mut Self::Context) -> Self::Result {
        let (mean, points_within_len, iterations) = self.mean_shift_single(msg.start_center, self.bandwidth);
        msg.source.do_send(MeanShiftHelperResponse { source: ctx.address().recipient(), mean, points_within_len, iterations }).unwrap();
    }
}

impl Handler<PoisonPill> for MeanShiftHelper {
    type Result = ();

    fn handle(&mut self, msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
