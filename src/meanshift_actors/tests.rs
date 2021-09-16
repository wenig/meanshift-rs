use ndarray::prelude::*;
use actix::prelude::*;
use crate::meanshift_actors::*;
use std::sync::{Arc, Mutex};
use crate::test_utils::read_data;

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


#[test]
fn test_runs_meanshift() {
    env_logger::init();

    let result = Arc::new(Mutex::new(None));
    let cloned_result = Arc::clone(&result);
    let labels = Arc::new(Mutex::new(None));
    let cloned_labels = Arc::clone(&labels);


    let _system = System::run(move || {
        let dataset = read_data("data/test.csv");

        let receiver = MeanShiftReceiver {result: cloned_result, labels: cloned_labels}.start();
        let meanshift = MeanShiftActor::new(8).start();
        meanshift.do_send(MeanShiftMessage { source: Some(receiver.recipient()), data: dataset });
    });

    let expects: Array2<f32> = arr2(&[
        [0.5185592, 0.43546146, 0.5697923]
    ]);
    let received = (*result.lock().unwrap()).as_ref().unwrap().clone();
    assert_eq!(expects[[0, 0]], received[[0, 0]]);
    assert_eq!(expects[[0, 1]], received[[0, 1]]);
    assert_eq!(expects[[0, 2]], received[[0, 2]]);

    let expected_label = 0;
    let mut received_label = (*labels.lock().unwrap()).as_ref().unwrap().clone();
    received_label.dedup();
    assert_eq!(expected_label, received_label[0])
}
