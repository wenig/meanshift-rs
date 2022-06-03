use crate::meanshift_actors::interface::MySink;
use crate::meanshift_actors::*;
use crate::meanshift_base::LibData;
use crate::test_utils::{close_l1, read_data};
use actix::io::SinkWrite;
use actix::prelude::*;
use anyhow::{Error, Result};
use ndarray::prelude::*;
use tokio::sync::mpsc;

type TestResult<A> = (Array2<A>, Vec<usize>);

struct MeanShiftReceiver<A: LibData> {
    sink: SinkWrite<TestResult<A>, MySink<TestResult<A>>>,
}

impl<A: LibData> MeanShiftReceiver<A> {
    pub fn new(sink: SinkWrite<TestResult<A>, MySink<TestResult<A>>>) -> Self {
        Self { sink }
    }

    pub fn start_new(sender: mpsc::UnboundedSender<TestResult<A>>) -> Addr<Self> {
        Self::create(move |ctx| {
            let sink = MySink::new(sender);
            Self::new(SinkWrite::new(sink, ctx))
        })
    }
}

impl<A: LibData> actix::io::WriteHandler<()> for MeanShiftReceiver<A>
where
    A: Unpin + 'static + Clone,
{
    fn finished(&mut self, _ctxt: &mut Self::Context) {
        System::current().stop();
    }
}

impl<A: LibData> Actor for MeanShiftReceiver<A> {
    type Context = Context<Self>;

    fn stopped(&mut self, _ctx: &mut Context<Self>) {
        System::current().stop();
    }
}

impl<A: LibData> Handler<ClusteringResponse<A>> for MeanShiftReceiver<A> {
    type Result = ();

    fn handle(&mut self, msg: ClusteringResponse<A>, _ctx: &mut Self::Context) -> Self::Result {
        self.sink
            .write((msg.cluster_centers, msg.labels))
            .expect("Writing to sink failed!");
    }
}

#[test]
fn test_runs_meanshift() {
    env_logger::init();

    let expects: Array2<f32> = arr2(&[
        [
            100.,
            100.,
            100.,
            100.,
            100.,
            100.,
            100.,
            100.,
            100.,
            100.,
        ],
        [
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        ],
        [
            -100.,
            -100.,
            -100.,
            -100.,
            -100.,
            -100.,
            -100.,
            -100.,
            -100.,
            -100.,
        ],
    ]);

    let expected_labels: Vec<usize> = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];

    let (cluster_centers, labels) = run_system().unwrap();

    close_l1(expects[[0, 0]], cluster_centers[[0, 0]], 2.0);
    close_l1(expects[[0, 1]], cluster_centers[[0, 1]], 2.0);
    close_l1(expects[[0, 2]], cluster_centers[[0, 2]], 2.0);

    assert_eq!(expected_labels, labels)
}

#[actix_rt::main]
async fn run_system<A: LibData>() -> Result<TestResult<A>> {
    let (sender, mut receiver) = mpsc::unbounded_channel();
    let dataset = read_data("data/cluster.csv");

    let ms_receiver = MeanShiftReceiver::start_new(sender);
    let meanshift = MeanShiftActor::new(8).start();
    meanshift.do_send(MeanShiftMessage {
        source: Some(ms_receiver.recipient()),
        data: dataset,
    });

    receiver
        .recv()
        .await
        .ok_or_else(|| Error::msg("No value received!"))
}
