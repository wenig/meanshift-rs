use actix::prelude::*;
use ndarray::{Array2, Array1};
use crate::ClusteringResponse;
use crate::meanshift_base::LibData;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct MeanShiftMessage<A: LibData> {
    pub source: Option<Recipient<ClusteringResponse<A>>>,
    pub data: Array2<A>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct MeanShiftHelperWorkMessage<A: LibData> {
    pub source: Recipient<MeanShiftHelperResponse<A>>,
    pub start_center: usize
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct MeanShiftHelperResponse<A: LibData> {
    pub source: Recipient<MeanShiftHelperWorkMessage<A>>,
    pub mean: Array1<A>,
    pub points_within_len: usize,
    pub iterations: usize
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct MeanShiftLabelHelperMessage {
    pub source: Recipient<MeanShiftLabelHelperResponse>,
    pub point_id: usize
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct MeanShiftLabelHelperResponse {
    pub source: Recipient<MeanShiftLabelHelperMessage>,
    pub point_id: usize,
    pub label: usize
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct PoisonPill;
