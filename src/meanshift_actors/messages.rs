use actix::prelude::*;
use ndarray::{Array2, Array1, ArcArray2};


#[derive(Message)]
#[rtype(Result = "()")]
pub struct MeanShiftMessage {
    pub source: Option<Recipient<MeanShiftResponse>>,
    pub data: Array2<f32>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct MeanShiftResponse {
    pub cluster_centers: Array2<f32>,
    pub labels: Vec<usize>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct MeanShiftHelperWorkMessage {
    pub source: Recipient<MeanShiftHelperResponse>,
    pub start_center: usize
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct MeanShiftHelperResponse {
    pub source: Recipient<MeanShiftHelperWorkMessage>,
    pub mean: Array1<f32>,
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
