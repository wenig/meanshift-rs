use actix::{Actor, SyncContext, Handler, ActorContext};
use crate::meanshift_actors::messages::{MeanShiftLabelHelperMessage, MeanShiftLabelHelperResponse, PoisonPill};
use ndarray::{ArcArray2};


use crate::meanshift_base::{closest_distance, DistanceMeasure, LibData};


pub struct MeanShiftLabelHelper<A> {
    data: ArcArray2<A>,
    distance_measure: DistanceMeasure,
    cluster_centers: ArcArray2<A>
}

impl<A: LibData> MeanShiftLabelHelper<A> {
    pub fn new(data: ArcArray2<A>, distance_measure: DistanceMeasure, cluster_centers: ArcArray2<A>) -> Self {
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

impl<A: LibData> Actor for MeanShiftLabelHelper<A> {
    type Context = SyncContext<Self>;
}

impl<A: LibData> Handler<MeanShiftLabelHelperMessage> for MeanShiftLabelHelper<A> {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftLabelHelperMessage, ctx: &mut Self::Context) -> Self::Result {
        let label = self.closest_distance(msg.point_id);
        msg.source.do_send(MeanShiftLabelHelperResponse { source: ctx.address().recipient(), point_id: msg.point_id, label }).unwrap();
    }
}

impl<A: LibData> Handler<PoisonPill> for MeanShiftLabelHelper<A> {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
