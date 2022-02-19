# MeanShift-rs

## Install

### 0. Minimum Requirements

- `rustc 1.54.0`
- `Python 3.8`

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

## Usage

### Python

```python
from meanshift_rs import MeanShift

# ... load data

ms = MeanShift()
ms.fit(data)

print(ms.cluster_centers)
print(ms.labels)
```
