use ndarray::{Array2, Array1};
use std::fs::File;
use std::io::{BufReader, BufRead};
use csv::{ReaderBuilder, Trim};
use std::str::FromStr;
use crate::meanshift_base::LibDataType;


pub fn read_data(file_path: &str) -> Array2<LibDataType> {
    let file = File::open(file_path).unwrap();
    let count_reader = BufReader::new(file);
    let n_lines = count_reader.lines().count() - 1;

    let file = File::open(file_path).unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).trim(Trim::All).from_reader(file);

    let n_rows = n_lines;
    let n_columns = reader.headers().unwrap().len();

    let flat_data: Array1<LibDataType> = reader.records().into_iter().flat_map(|rec| {
        rec.unwrap().iter().map(|b| {
            LibDataType::from_str(b).unwrap()
        }).collect::<Vec<LibDataType>>()
    }).collect();

    flat_data.into_shape((n_rows, n_columns)).expect("Could not deserialize sent data")
}
