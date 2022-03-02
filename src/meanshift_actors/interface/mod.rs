#[cfg(test)]
mod tests;
mod sink;
mod actor;

use anyhow::{Result, Error};
use crate::interface::{MeanShiftInterface, Parameters, MeanShiftResult};
use crate::meanshift_base::{LibData, MeanShiftBase};
pub use crate::meanshift_actors::interface::sink::MySink;
use sorted_vec::SortedVec;
use ndarray::Array2;
use crate::meanshift_actors::interface::actor::SinkActor;
use actix::{Actor, Handler};
use actix::io::SinkWrite;
use tokio::sync::mpsc;
use crate::{ClusteringResponse, MeanShiftActor};
use crate::meanshift_actors::MeanShiftMessage;


impl<A: LibData> MeanShiftInterface<A> for MeanShiftActor<A>  {
    fn init(parameters: Parameters<A>) -> Self {
        Self {
            meanshift: MeanShiftBase {
                bandwidth: parameters.bandwidth,
                distance_measure: parameters.distance_measure,
                ..Default::default()
            },
            helpers: None,
            label_helpers: None,
            n_threads: parameters.n_threads,
            receiver: None,
            centers_sent: 0,
            distances_sent: 0,
            start_time: None,
            labels: SortedVec::new()
        }
    }

    fn fit(&mut self, data: Array2<A>) -> Result<MeanShiftResult<A>> {
        let actor = self.clone();
        actix_rt::System::new().block_on(async move {
            actor_fit(actor, data).await
        })
    }
}


impl<A: LibData> Handler<ClusteringResponse<A>> for SinkActor<MeanShiftResult<A>> {
    type Result = ();

    fn handle(&mut self, msg: ClusteringResponse<A>, _ctx: &mut Self::Context) -> Self::Result {
        let _ = self.sink.write((msg.cluster_centers, msg.labels));
        self.sink.close()
    }
}


async fn actor_fit<A: LibData>(actor: MeanShiftActor<A>, data: Array2<A>) -> Result<MeanShiftResult<A>>  {
    let (sender, mut receiver) = mpsc::unbounded_channel();

    let sink_actor = SinkActor::create(move |ctx| {
        let sink = MySink::new(sender);
        SinkActor::new(SinkWrite::new(sink, ctx))
    });

    let addr = actor.start();
    addr.do_send(MeanShiftMessage {
        source: Some(sink_actor.recipient()),
        data
    });

    if let Some(r) = receiver.recv().await{
        Ok(r)
    } else {
        Err(Error::msg("Await resulted in None value!"))
    }
}
