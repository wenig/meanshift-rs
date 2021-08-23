use actix::{Actor, SyncContext, Handler, ActorContext};
use crate::meanshift::messages::{MeanShiftHelperWorkMessage, MeanShiftLabelHelperMessage, MeanShiftLabelHelperResponse, PoisonPill};

use ndarray::prelude::*;
use kdtree::{KdTree};
use kdtree::distance::squared_euclidean;

use ndarray::{ArcArray2};

use crate::meanshift::{RefArray, MeanShiftHelperResponse, DistanceMeasure};
use std::sync::Arc;
use actix::dev::MessageResponse;


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
        let point = self.data.select(Axis(0), &[point_id]).mean_axis(Axis(0)).unwrap();
        self.cluster_centers.axis_iter(Axis(0)).map(|center| {
            self.distance_measure.call()(point.as_slice().unwrap(), center.as_slice().unwrap())
        }).enumerate().reduce(|(min_i, min), (i, x)| {
            if x < min { (i, x) } else { (min_i, min) }
        }).unwrap().0
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
