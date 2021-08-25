use ndarray::{Array2, Array1};
use std::fs::File;
use std::io::{BufReader, BufRead};
use csv::{ReaderBuilder, Trim};
use std::str::FromStr;


pub fn read_data(file_path: &str) -> Array2<f32> {
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
