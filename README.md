<div align="center">

# MeanShift-rs
[![pipeline status](https://gitlab.hpi.de/phillip.wenig/meanshift-rs/badges/main/pipeline.svg)](https://gitlab.hpi.de/phillip.wenig/meanshift-rs/-/commits/main)
[![coverage report](https://gitlab.hpi.de/phillip.wenig/meanshift-rs/badges/main/coverage.svg)](https://gitlab.hpi.de/phillip.wenig/meanshift-rs/-/commits/main)


[![Git tag](https://img.shields.io/badge/tag-0.9.3-blue?style=for-the-badge)](./Cargo.toml)
</div>

## Install as Python Package

### 0. Minimum Requirements

- `rustc 1.54.0`
- `Python 3.7`

### 1. Create Python Environment

```bash
python3 -m venv .venv
```

### 2. Install MeanShift-rs

```bash
make install
```

or

```bash
pip install -r requirements.txt
bash ./tasks.sh release-install
```

or

```bash
pip install meanshift-rs
```

## Usage

### Rust

```rust
use meanshift_rs::{MeanShiftActor, Parameters};

fn example_fn(dataset: Array2<f64>) {
    let parameters = Parameters::default();
    let mut mean_shift = MeanShiftActor::init(parameters);
    let (centers, labels) = mean_shift.fit(dataset)
        .expect("No MeanShiftResult was returned!");
}
```

### Python

```python
from meanshift_rs import MeanShift

# ... load data

ms = MeanShift()
ms.fit(data)

print(ms.cluster_centers)
print(ms.labels)
```
