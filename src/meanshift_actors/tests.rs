use ndarray::prelude::*;
use actix::prelude::*;
use crate::meanshift::*;
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::str::FromStr;
use csv::{ReaderBuilder, Trim};
use log::*;

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

pub fn read_data_(file_path: &str) -> Array2<f32> {
    let file = File::open(file_path).unwrap();
    let count_reader = BufReader::new(file);
    let n_lines = count_reader.lines().count() - 1;

    let file = File::open(file_path).unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).trim(Trim::All).from_reader(file);

    let n_rows = n_lines;
    let n_columns = reader.headers().unwrap().len();

    let flat_data: Array1<f32> = reader.records().into_iter().flat_map(|rec| {
        rec.unwrap().iter().map(|b| {
            f32::from_str(b).unwrap()
        }).collect::<Vec<f32>>()
    }).collect();

    flat_data.into_shape((n_rows, n_columns)).expect("Could not deserialize sent data")
}

#[test]
fn test_runs_meanshift() {
    env_logger::init();

    debug!("test started");

    let result = Arc::new(Mutex::new(None));
    let cloned_result = Arc::clone(&result);
    let labels = Arc::new(Mutex::new(None));
    let cloned_labels = Arc::clone(&labels);


    let _system = System::run(move || {
        let dataset = read_data_("data/test.csv");

        let receiver = MeanShiftReceiver {result: cloned_result, labels: cloned_labels}.start();
        let meanshift = MeanShiftActor::new(20).start();
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
    debug!("received labels {:?}", received_label);
    assert_eq!(expected_label, received_label[0])
}
