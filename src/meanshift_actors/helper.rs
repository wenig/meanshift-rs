use actix::{Actor, SyncContext, Handler, ActorContext};
use crate::meanshift_actors::messages::{MeanShiftHelperWorkMessage, PoisonPill};

use ndarray::prelude::*;
use kdtree::{KdTree};
use ndarray::{ArcArray2};

use crate::meanshift_actors::{MeanShiftHelperResponse};
use std::sync::Arc;
use crate::meanshift_base::{mean_shift_single, RefArray, DistanceMeasure, LibData};


pub struct MeanShiftHelper<A: LibData>
{
    data: ArcArray2<A>,
    tree: Arc<KdTree<A, usize, RefArray<A>>>,
    bandwidth: A,
    distance_measure: DistanceMeasure
}

impl<A: LibData> MeanShiftHelper<A>  {
    pub fn new(data: ArcArray2<A>, tree: Arc<KdTree<A, usize, RefArray<A>>>, bandwidth: A, distance_measure: DistanceMeasure) -> Self {
        Self {
            data,
            tree,
            bandwidth,
            distance_measure
        }
    }

    fn mean_shift_single(&mut self, seed: usize, bandwidth: A) -> (Array1<A>, usize, usize) {
        mean_shift_single(
            self.data.to_shared(),
            self.tree.clone(),
            seed,
            bandwidth,
            self.distance_measure.clone()
        )
    }
}

impl<A: LibData> Actor for MeanShiftHelper<A>  {
    type Context = SyncContext<Self>;
}

impl<A: LibData> Handler<MeanShiftHelperWorkMessage<A>> for MeanShiftHelper<A>  {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftHelperWorkMessage<A>, ctx: &mut Self::Context) -> Self::Result {
        let (mean, points_within_len, iterations) = self.mean_shift_single(msg.start_center, self.bandwidth);
        msg.source.do_send(MeanShiftHelperResponse { source: ctx.address().recipient(), mean, points_within_len, iterations }).unwrap();
    }
}

impl<A: LibData> Handler<PoisonPill> for MeanShiftHelper<A>  {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
