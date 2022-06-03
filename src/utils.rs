use actix::Message;
use ndarray::Array2;

#[derive(Message)]
#[rtype(Result = "()")]
pub struct ClusteringResponse<A> {
    pub cluster_centers: Array2<A>,
    pub labels: Vec<usize>,
}
