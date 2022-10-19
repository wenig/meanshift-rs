use crate::utils::LibData;
use csv::{ReaderBuilder, Trim};
use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub(crate) fn close_l1<A: LibData>(a: A, b: A, delta: A) {
    assert!((a - b).abs() < delta)
}

pub(crate) fn read_data<A: LibData>(file_path: &str) -> Array2<A> {
    let file = File::open(file_path).unwrap();
    let count_reader = BufReader::new(file);
    let n_lines = count_reader.lines().count() - 1;

    let file = File::open(file_path).unwrap();
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .trim(Trim::All)
        .from_reader(file);

    let n_rows = n_lines;
    let n_columns = reader.headers().unwrap().len();

    let flat_data: Array1<A> = reader
        .records()
        .into_iter()
        .flat_map(|rec| {
            rec.unwrap()
                .iter()
                .map(|b| A::from_str(b).ok().unwrap())
                .collect::<Vec<A>>()
        })
        .collect();

    flat_data
        .into_shape((n_rows, n_columns))
        .expect("Could not deserialize sent data")
}
