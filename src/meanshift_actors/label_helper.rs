use actix::{Actor, SyncContext, Handler, ActorContext};
use crate::meanshift_actors::messages::{MeanShiftHelperWorkMessage, MeanShiftLabelHelperMessage, MeanShiftLabelHelperResponse, PoisonPill};

use ndarray::prelude::*;
use kdtree::{KdTree};
use kdtree::distance::squared_euclidean;

use ndarray::{ArcArray2};

use crate::meanshift_actors::{MeanShiftHelperResponse};
use std::sync::Arc;
use actix::dev::MessageResponse;
use crate::meanshift_base::{closest_distance, DistanceMeasure};


pub struct MeanShiftLabelHelper {
    data: ArcArray2<f32>,
    distance_measure: DistanceMeasure,
    cluster_centers: ArcArray2<f32>
}

impl MeanShiftLabelHelper {
    pub fn new(data: ArcArray2<f32>, distance_measure: DistanceMeasure, cluster_centers: ArcArray2<f32>) -> Self {
        Self {
            data,
            distance_measure,
            cluster_centers
        }
    }

    fn closest_distance(&mut self, point_id: usize) -> usize {
        closest_distance(
            self.data.to_shared(),
            point_id,
            self.cluster_centers.to_shared(),
            self.distance_measure.clone()
        )
    }
}

impl Actor for MeanShiftLabelHelper {
    type Context = SyncContext<Self>;
}

impl Handler<MeanShiftLabelHelperMessage> for MeanShiftLabelHelper {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftLabelHelperMessage, ctx: &mut Self::Context) -> Self::Result {
        let label = self.closest_distance(msg.point_id);
        msg.source.do_send(MeanShiftLabelHelperResponse { source: ctx.address().recipient(), point_id: msg.point_id, label }).unwrap();
    }
}

impl Handler<PoisonPill> for MeanShiftLabelHelper {
    type Result = ();

    fn handle(&mut self, msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
