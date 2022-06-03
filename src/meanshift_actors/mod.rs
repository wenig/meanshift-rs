mod helper;
mod interface;
mod label_helper;
mod messages;
#[cfg(test)]
mod tests;

use crate::meanshift_actors::helper::MeanShiftHelper;
use crate::meanshift_actors::label_helper::MeanShiftLabelHelper;
pub use crate::meanshift_actors::messages::{
    MeanShiftHelperResponse, MeanShiftHelperWorkMessage, MeanShiftMessage,
};
use crate::meanshift_actors::messages::{
    MeanShiftLabelHelperMessage, MeanShiftLabelHelperResponse, PoisonPill,
};
use crate::meanshift_base::{LibData, MeanShiftBase};
use crate::utils::ClusteringResponse;
use actix::{Actor, ActorContext, Addr, AsyncContext, Context, Handler, Recipient, SyncArbiter};
use log::*;
use ndarray::Array1;
use sorted_vec::SortedVec;
use std::cmp::Ordering;
use std::time::SystemTime;

#[derive(Debug, Clone)]
struct SortedElement {
    key: usize,
    value: usize,
}

impl SortedElement {
    pub fn new(key: usize, value: usize) -> Self {
        Self { key, value }
    }
}

impl Eq for SortedElement {}

impl PartialEq<Self> for SortedElement {
    fn eq(&self, other: &Self) -> bool {
        self.key.eq(&other.key)
    }
}

impl PartialOrd<Self> for SortedElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.key.partial_cmp(&other.key)
    }
}

impl Ord for SortedElement {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

#[derive(Clone)]
pub struct MeanShiftActor<A: LibData> {
    meanshift: MeanShiftBase<A>,
    helpers: Option<Addr<MeanShiftHelper<A>>>,
    label_helpers: Option<Addr<MeanShiftLabelHelper<A>>>,
    n_threads: usize,
    receiver: Option<Recipient<ClusteringResponse<A>>>,
    centers_sent: usize,
    distances_sent: usize,
    #[allow(dead_code)]
    start_time: Option<SystemTime>,
    labels: SortedVec<SortedElement>,
}

impl<A: LibData> Actor for MeanShiftActor<A> {
    type Context = Context<Self>;
}

impl<A: LibData> MeanShiftActor<A> {
    pub fn new(n_threads: usize) -> Self {
        Self {
            meanshift: Default::default(),
            helpers: None,
            label_helpers: None,
            n_threads,
            receiver: None,
            centers_sent: 0,
            distances_sent: 0,
            start_time: None,
            labels: SortedVec::new(),
        }
    }

    fn create_helpers(&mut self) {
        let data = self.meanshift.dataset.as_ref().unwrap().to_shared();
        let tree = self.meanshift.tree.as_ref().unwrap().clone();
        let bandwidth = *self
            .meanshift
            .bandwidth
            .as_ref()
            .expect("You must estimate or give a bandwidth before starting the algorithm!");
        let distance_measure = self.meanshift.distance_measure.clone();

        self.helpers = Some(SyncArbiter::start(self.n_threads, move || {
            MeanShiftHelper::new(
                data.clone(),
                tree.clone(),
                bandwidth,
                distance_measure.clone(),
            )
        }));
    }

    fn distribute_data(&mut self, rec: Recipient<MeanShiftHelperResponse<A>>) {
        self.meanshift.estimate_bandwidth();
        self.meanshift.build_center_tree();
        self.create_helpers();

        let n_clusters = self
            .n_threads
            .min(self.meanshift.dataset.as_ref().unwrap().shape()[0]);

        self.centers_sent = n_clusters;
        for t in 0..n_clusters {
            self.helpers
                .as_ref()
                .unwrap()
                .do_send(MeanShiftHelperWorkMessage {
                    source: rec.clone(),
                    start_center: t,
                });
        }
    }

    fn add_mean(&mut self, mean: Array1<A>, points_within_len: usize, iterations: usize) {
        if points_within_len > 0 {
            let identifier = self.meanshift.means.len();
            self.meanshift
                .means
                .push((mean, points_within_len, iterations, identifier));
        }
    }

    fn collect_means(&mut self, rec: Recipient<MeanShiftLabelHelperResponse>) {
        self.meanshift.collect_means();

        let arc_cluster_centers = self.meanshift.cluster_centers.as_ref().unwrap().to_shared();
        let data = self
            .meanshift
            .dataset
            .as_ref()
            .unwrap()
            .clone()
            .into_shared();
        let distance_measure = self.meanshift.distance_measure.clone();

        self.label_helpers = Some(SyncArbiter::start(self.n_threads, move || {
            MeanShiftLabelHelper::new(
                data.clone(),
                distance_measure.clone(),
                arc_cluster_centers.clone(),
            )
        }));

        self.distribute_distance_calculation(rec)
    }

    fn distribute_distance_calculation(&mut self, rec: Recipient<MeanShiftLabelHelperResponse>) {
        let n_clusters = self
            .n_threads
            .min(self.meanshift.dataset.as_ref().unwrap().shape()[0]);
        self.distances_sent = n_clusters;
        for t in 0..n_clusters {
            self.label_helpers
                .as_ref()
                .unwrap()
                .do_send(MeanShiftLabelHelperMessage {
                    source: rec.clone(),
                    point_id: t,
                });
        }
    }
}

impl<A: LibData> Handler<MeanShiftMessage<A>> for MeanShiftActor<A> {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftMessage<A>, ctx: &mut Self::Context) -> Self::Result {
        if self.meanshift.dataset.is_none() {
            self.meanshift.dataset = Some(msg.data);
            self.receiver = msg.source;
            self.distribute_data(ctx.address().recipient())
        }
    }
}

impl<A: LibData> Handler<MeanShiftHelperResponse<A>> for MeanShiftActor<A> {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftHelperResponse<A>, ctx: &mut Self::Context) -> Self::Result {
        self.add_mean(msg.mean, msg.points_within_len, msg.iterations);

        if self.meanshift.means.len() == self.meanshift.dataset.as_ref().unwrap().shape()[0] {
            debug!("all means received");
            self.helpers.as_ref().unwrap().do_send(PoisonPill);
            self.collect_means(ctx.address().recipient());
            debug!("cluster_centers {:?}", self.meanshift.cluster_centers);
        } else if self.centers_sent < self.meanshift.dataset.as_ref().unwrap().shape()[0] {
            let start_center = self.centers_sent;
            self.centers_sent += 1;
            msg.source
                .do_send(MeanShiftHelperWorkMessage {
                    source: ctx.address().recipient(),
                    start_center,
                })
                .unwrap();
        }
    }
}

impl<A: LibData> Handler<MeanShiftLabelHelperResponse> for MeanShiftActor<A> {
    type Result = ();

    fn handle(
        &mut self,
        msg: MeanShiftLabelHelperResponse,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        self.labels
            .insert(SortedElement::new(msg.point_id, msg.label));

        let len = self.meanshift.dataset.as_ref().unwrap().shape()[0];
        if self.labels.len() == len {
            self.label_helpers.as_ref().unwrap().do_send(PoisonPill);
            match &self.receiver {
                Some(recipient) => {
                    let cluster_centers =
                        (*self.meanshift.cluster_centers.as_ref().unwrap()).clone();
                    let labels = self.labels.iter().map(|x| x.value).collect();
                    recipient
                        .do_send(ClusteringResponse {
                            cluster_centers,
                            labels,
                        })
                        .unwrap();
                }
                None => (),
            }
            ctx.stop();
        } else if self.distances_sent < len {
            let point_id = self.distances_sent;
            self.distances_sent += 1;
            msg.source
                .do_send(MeanShiftLabelHelperMessage {
                    source: ctx.address().recipient(),
                    point_id,
                })
                .unwrap();
        }
    }
}
