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

    fn handle(&mut self, msg: ClusteringResponse<A>, ctx: &mut Self::Context) -> Self::Result {
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
            1.26765306e+01,
            2.56051020e+00,
            2.34071429e+00,
            2.06602040e+01,
            9.50000000e+01,
            2.01673469e+00,
            1.53959184e+00,
            4.02857143e-01,
            1.40938776e+00,
            5.00591836e+00,
            9.01734694e-01,
            2.30122449e+00,
            5.62153061e+02,
        ],
        [
            1.34051111e+01,
            2.32866667e+00,
            2.40444444e+00,
            1.84844444e+01,
            1.08555556e+02,
            2.48600000e+00,
            2.28955556e+00,
            3.19777778e-01,
            1.69977778e+00,
            5.19977778e+00,
            9.63466667e-01,
            2.88466667e+00,
            9.15177778e+02,
        ],
        [
            1.37433333e+01,
            1.89538462e+00,
            2.42179487e+00,
            1.71743589e+01,
            1.04384615e+02,
            2.80589744e+00,
            2.94512821e+00,
            2.84871795e-01,
            1.85948718e+00,
            5.45333333e+00,
            1.07256410e+00,
            3.11128205e+00,
            1.14230769e+03,
        ],
    ]);

    let expected_labels: Vec<usize> = vec![
        2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2,
        2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 0, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
    ];

    let (cluster_centers, labels) = run_system().unwrap();

    close_l1(expects[[0, 0]], cluster_centers[[0, 0]], 0.01);
    close_l1(expects[[0, 1]], cluster_centers[[0, 1]], 0.01);
    close_l1(expects[[0, 2]], cluster_centers[[0, 2]], 0.01);

    assert_eq!(expected_labels, labels)
}

#[actix_rt::main]
async fn run_system<A: LibData>() -> Result<TestResult<A>> {
    let (sender, mut receiver) = mpsc::unbounded_channel();
    let dataset = read_data("data/wine.csv");

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
