#[cfg(test)]
mod tests;
mod sink;
mod actor;

use anyhow::{Result, Error};
use crate::interface::{MeanShiftInterface, Parameters, MeanShiftResult};
use crate::meanshift_base::MeanShiftBase;
pub use crate::meanshift_actors::interface::sink::MySink;
use sorted_vec::SortedVec;
use ndarray::Array2;
use crate::meanshift_actors::interface::actor::SinkActor;
use actix::{Actor, Addr, Handler, ContextFutureSpawner};
use actix::io::SinkWrite;
use tokio::sync::mpsc;
use crate::MeanShiftActor;
use crate::meanshift_actors::{MeanShiftResponse, MeanShiftMessage};

impl MeanShiftInterface for MeanShiftActor {
    fn init(parameters: Parameters) -> Self {
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

    fn fit(&mut self, data: Array2<f32>) -> Result<MeanShiftResult> {
        let actor = self.clone();
        actix_rt::System::new().block_on(async move {
            actor_fit(actor, data).await
        })
    }
}


impl Handler<MeanShiftResponse> for SinkActor<MeanShiftResult> {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftResponse, _ctx: &mut Self::Context) -> Self::Result {
        let _ = self.sink.write((msg.cluster_centers, msg.labels));
        self.sink.close()
    }
}


async fn actor_fit(actor: MeanShiftActor, data: Array2<f32>) -> Result<MeanShiftResult> {
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
