use ndarray::prelude::*;
use actix::prelude::*;
use crate::meanshift_actors::*;
use std::sync::{Arc, Mutex};
use crate::test_utils::read_data;
use tokio::time::{Duration};
use actix_rt::time::sleep;

struct MeanShiftReceiver {
    result: Arc<Mutex<Option<Array2<f32>>>>,
    labels: Arc<Mutex<Option<Vec<usize>>>>
}

impl Actor for MeanShiftReceiver {
    type Context = Context<Self>;

    fn stopped(&mut self, _ctx: &mut Context<Self>) {
        System::current().stop();
    }
}

impl Handler<MeanShiftResponse> for MeanShiftReceiver {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftResponse, ctx: &mut Self::Context) -> Self::Result {
        *(self.result.lock().unwrap()) = Some(msg.cluster_centers);
        *(self.labels.lock().unwrap()) = Some(msg.labels);
        ctx.stop();
    }
}


fn close_l1(a: f32, b: f32, delta: f32) -> bool {
    (a - b).abs() < delta
}


#[test]
fn test_runs_meanshift() {
    env_logger::init();

    let result = Arc::new(Mutex::new(None));
    let cloned_result = Arc::clone(&result);
    let labels = Arc::new(Mutex::new(None));
    let cloned_labels = Arc::clone(&labels);

    run_system(cloned_result, cloned_labels);

    let expects: Array2<f32> = arr2(&[
        [1.26765306e+01, 2.56051020e+00, 2.34071429e+00, 2.06602040e+01,
        9.50000000e+01, 2.01673469e+00, 1.53959184e+00, 4.02857143e-01,
        1.40938776e+00, 5.00591836e+00, 9.01734694e-01, 2.30122449e+00,
        5.62153061e+02],
       [1.34051111e+01, 2.32866667e+00, 2.40444444e+00, 1.84844444e+01,
        1.08555556e+02, 2.48600000e+00, 2.28955556e+00, 3.19777778e-01,
        1.69977778e+00, 5.19977778e+00, 9.63466667e-01, 2.88466667e+00,
        9.15177778e+02],
       [1.37433333e+01, 1.89538462e+00, 2.42179487e+00, 1.71743589e+01,
        1.04384615e+02, 2.80589744e+00, 2.94512821e+00, 2.84871795e-01,
        1.85948718e+00, 5.45333333e+00, 1.07256410e+00, 3.11128205e+00,
        1.14230769e+03]
    ]);

    let expected_labels: Vec<usize> = vec![
        2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1,
        2, 1, 1, 1, 2, 2, 0, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
        1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0];

    let received = (*result.lock().unwrap()).as_ref().unwrap().clone();
    assert!(close_l1(expects[[0, 0]], received[[0, 0]], 0.01));
    assert!(close_l1(expects[[0, 1]], received[[0, 1]], 0.01));
    assert!(close_l1(expects[[0, 2]], received[[0, 2]], 0.01));

    let received_labels = (*labels.lock().unwrap()).as_ref().unwrap().clone();
    assert_eq!(expected_labels, received_labels)
}

#[actix_rt::main]
async fn run_system(cloned_result: Arc<Mutex<Option<Array2<f32>>>>, cloned_labels: Arc<Mutex<Option<Vec<usize>>>>) {
    let dataset = read_data("data/wine.csv");

    let receiver = MeanShiftReceiver {result: cloned_result, labels: cloned_labels}.start();
    let meanshift = MeanShiftActor::new(8).start();
    meanshift.do_send(MeanShiftMessage { source: Some(receiver.recipient()), data: dataset });
    sleep(Duration::from_millis(500)).await;
}
